"""Pluggable preview storage backend for the TTS voice catalog (GCS or local).

Parallel to `app.services.gcp.storage.upload_file_to_gcs` / `GCSStorage` —
deliberately not reusing `GCSStorage` here. That class hardcodes its bucket
to the global `GCS_BUCKET` (the recordings bucket, default "atoms-sdk") with
no constructor param to target a different one, and `upload_file_to_gcs`
only ever returns a URL for that same bucket. Previews need their own
bucket (`TTS_PREVIEW_GCS_BUCKET`), so this module calls `get_gcs_bucket()`
directly — already bucket-name-parameterized and unmodified — instead of
touching any existing storage code.

Path shape: `tts-previews/{provider}/{voice_id}/{language}-{key}.wav`.

Besides the WAVs themselves, this module owns the **preview manifest** —
`tts-previews/manifest.json`, stored in the same backend. It maps
"provider/voice_id" to that voice's preview entries
(`{language, url, content_key, format}` or `{language, error, content_key}`)
and is the only mutable state in the catalog system: the voice list itself
is static (app/ai/voice/tts/catalog.json) and the manifest records which
previews have been generated for it. Reads are TTL-cached in-process
(matching the endpoint's 5-minute Cache-Control); writes always
read-modify-write fresh, and a load failure during a write aborts rather
than risking overwriting a good manifest with a partial one. The manifest
lives in the public preview location by design — it contains only public
preview URLs and content-key hashes.
"""

import asyncio
import json
import os
import re
import tempfile
import time
from typing import Optional

from google.api_core.exceptions import NotFound

from app.core.config.static import (
    TTS_PREVIEW_GCS_BUCKET,
    TTS_PREVIEW_PUBLIC_BASE_URL,
    TTS_PREVIEW_STORAGE,
)
from app.core.logger import logger
from app.services.gcp.storage.client import get_gcs_bucket

__all__ = [
    "store_preview",
    "load_manifest",
    "load_manifest_fresh",
    "update_previews",
    "LOCAL_PREVIEW_DIR",
    "MANIFEST_NAME",
]

# Module-level names (not read from app.core.config.static at call time) so
# tests can monkeypatch them directly, e.g. `monkeypatch.setattr(
# preview_storage, "TTS_PREVIEW_STORAGE", "local")`.
LOCAL_PREVIEW_DIR = os.path.join("var", "tts-previews")

MANIFEST_NAME = "manifest.json"

# Matches the endpoint's `Cache-Control: max-age=300`: a manifest written by
# another pod becomes visible here within the same window clients already
# tolerate for staleness.
MANIFEST_CACHE_TTL_SECS = 300.0

# Read/written only from coroutines on the single asyncio event loop (never
# inside the asyncio.to_thread closures below), so cooperative scheduling makes
# a lock unnecessary; concurrent writers are last-write-wins (see
# update_previews).
_manifest_cache: Optional[tuple[float, dict]] = None

# Voice IDs are UUIDs/base62/speaker names, languages are BCP-47-ish, keys
# are hex — none of those legitimately need "/", "\", or "..". Reject
# anything else so these values (which flow straight into filesystem paths
# and GCS blob names) can't escape their directory.
_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+")


def _validate_component(name: str, value: str) -> str:
    if not value or not _COMPONENT_RE.fullmatch(value) or ".." in value:
        raise ValueError(
            f"Invalid preview path component {name}={value!r}: must be "
            "non-empty, match [A-Za-z0-9._-]+, and not contain '..'"
        )
    return value


def _rel_path(provider: str, voice_id: str, language: str, key: str) -> str:
    provider = _validate_component("provider", provider)
    voice_id = _validate_component("voice_id", voice_id)
    language = _validate_component("language", language)
    key = _validate_component("key", key)
    return f"{provider}/{voice_id}/{language}-{key}.wav"


def _public_url(rel: str) -> str:
    # Root-relative URLs only resolve when the caller shares this API's
    # origin. That's false for split-origin deployments (e.g. a frontend
    # dev proxy that only forwards specific path prefixes, or any setup
    # where the API isn't served from "/") — those need an absolute URL.
    # Prefix with TTS_PREVIEW_PUBLIC_BASE_URL when set; fall back to the
    # root-relative form for same-origin setups where it's unset.
    if TTS_PREVIEW_PUBLIC_BASE_URL:
        return f"{TTS_PREVIEW_PUBLIC_BASE_URL.rstrip('/')}/tts-previews/{rel}"
    return f"/tts-previews/{rel}"


async def store_preview(
    provider: str, voice_id: str, language: str, key: str, wav: bytes
) -> str:
    """Persist a preview WAV and return its public URL."""
    rel = _rel_path(provider, voice_id, language, key)
    if TTS_PREVIEW_STORAGE == "gcs":
        return await _store_gcs(rel, wav, content_type="audio/wav")
    return await _store_local(rel, wav)


async def _store_gcs(rel: str, data: bytes, content_type: str) -> str:
    destination_path = f"tts-previews/{rel}"

    def _upload() -> None:
        bucket = get_gcs_bucket(TTS_PREVIEW_GCS_BUCKET)
        if not bucket:
            raise RuntimeError(
                f"Failed to access GCS bucket '{TTS_PREVIEW_GCS_BUCKET}' for preview upload"
            )
        blob = bucket.blob(destination_path)
        blob.upload_from_string(data, content_type=content_type)

    await asyncio.to_thread(_upload)
    logger.info(f"Preview uploaded to gs://{TTS_PREVIEW_GCS_BUCKET}/{destination_path}")
    return _public_url(rel)


async def _store_local(rel: str, data: bytes) -> str:
    dest = os.path.join(LOCAL_PREVIEW_DIR, rel)

    def _write() -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Write-then-replace so concurrent static requests (this directory is
        # served at /tts-previews) never observe a partially written file.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".tmp")
        try:
            try:
                f = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen never took ownership, so the raw descriptor must be
                # closed here; once the `with` owns it, closing again would
                # hit an unrelated reused fd.
                os.close(fd)
                raise
            with f:
                f.write(data)
            os.replace(tmp, dest)
        except BaseException:
            os.unlink(tmp)
            raise

    await asyncio.to_thread(_write)
    return _public_url(rel)


# ---------------------------------------------------------------------------
# Preview manifest
# ---------------------------------------------------------------------------


def manifest_key(provider: str, voice_id: str) -> str:
    """Manifest map key for one voice."""
    return f"{provider}/{voice_id}"


async def load_manifest() -> dict:
    """Read the preview manifest, TTL-cached per process.

    Fails soft to the last cached value (or {}) on backend errors — the
    catalog endpoint should degrade to voices-without-previews rather than
    500 when storage hiccups. Writers never use this; see `update_previews`.
    """
    global _manifest_cache
    now = time.monotonic()
    if _manifest_cache and now - _manifest_cache[0] < MANIFEST_CACHE_TTL_SECS:
        return _manifest_cache[1]
    try:
        data = await load_manifest_fresh()
    except Exception:
        logger.exception("tts previews: manifest load failed")
        return _manifest_cache[1] if _manifest_cache else {}
    _manifest_cache = (now, data)
    return data


async def load_manifest_fresh() -> dict:
    """Read the manifest from the backend, bypassing the cache.

    Raises on backend errors so read-modify-write callers abort instead of
    clobbering good state with a partial manifest. A *missing* manifest is
    just {}; a manifest that exists but doesn't parse is treated as corrupt:
    logged loudly and returned as {} so the next successful write repairs it
    (self-healing), rather than wedging every future write behind the same
    parse error.
    """
    if TTS_PREVIEW_STORAGE == "gcs":
        return await _load_manifest_gcs()
    return await _load_manifest_local()


def _parse_manifest(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except ValueError:
        logger.exception(
            "tts previews: manifest.json is corrupt — treating as empty; "
            "the next successful reconcile write will replace it"
        )
        return {}
    if not isinstance(data, dict):
        logger.error(
            "tts previews: manifest.json root is not an object — treating as "
            "empty; the next successful reconcile write will replace it"
        )
        return {}
    return data


async def _load_manifest_local() -> dict:
    path = os.path.join(LOCAL_PREVIEW_DIR, MANIFEST_NAME)

    def _read() -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return _parse_manifest(f.read())

    return await asyncio.to_thread(_read)


async def _load_manifest_gcs() -> dict:
    destination_path = f"tts-previews/{MANIFEST_NAME}"

    def _download() -> dict:
        bucket = get_gcs_bucket(TTS_PREVIEW_GCS_BUCKET)
        if not bucket:
            raise RuntimeError(
                f"Failed to access GCS bucket '{TTS_PREVIEW_GCS_BUCKET}' for manifest read"
            )
        blob = bucket.blob(destination_path)
        if not blob.exists():
            return {}
        try:
            raw = blob.download_as_bytes()
        except NotFound:
            # Deleted between exists() and download — same as never existing.
            return {}
        return _parse_manifest(raw.decode("utf-8"))

    return await asyncio.to_thread(_download)


async def _store_manifest(manifest: dict) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    if TTS_PREVIEW_STORAGE == "gcs":
        await _store_gcs(MANIFEST_NAME, payload, content_type="application/json")
    else:
        await _store_local(MANIFEST_NAME, payload)


async def update_previews(provider: str, voice_id: str, previews: list) -> None:
    """Replace one voice's preview entries and persist the manifest.

    Fresh read-modify-write (never through the TTL cache) so sequential
    updates within one reconcile run compose; the cache is refreshed on
    success so the writing pod serves the new state immediately. Concurrent
    reconciles on different pods are last-write-wins per manifest — an
    accepted trade for an admin-only, idempotent operation.
    """
    global _manifest_cache
    manifest = dict(await load_manifest_fresh())
    manifest[manifest_key(provider, voice_id)] = previews
    await _store_manifest(manifest)
    _manifest_cache = (time.monotonic(), manifest)
