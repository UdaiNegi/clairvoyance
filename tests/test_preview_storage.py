import pytest

from app.services import preview_storage


@pytest.mark.asyncio
async def test_local_store_returns_served_url(tmp_path, monkeypatch):
    """Same-origin setups (TTS_PREVIEW_PUBLIC_BASE_URL unset) get a
    root-relative URL — resolves fine when the frontend and API share an
    origin."""
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_STORAGE", "local")
    monkeypatch.setattr(preview_storage, "LOCAL_PREVIEW_DIR", str(tmp_path))
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_PUBLIC_BASE_URL", "")
    url = await preview_storage.store_preview(
        "cartesia", "v1", "en", "k123", b"RIFFdata"
    )
    assert url == "/tts-previews/cartesia/v1/en-k123.wav"
    assert (tmp_path / "cartesia" / "v1" / "en-k123.wav").read_bytes() == b"RIFFdata"


@pytest.mark.asyncio
async def test_local_store_absolutizes_url_when_public_base_url_set(
    tmp_path, monkeypatch
):
    """Split-origin setups (e.g. a frontend dev proxy that only forwards
    specific path prefixes, or any deployment where the API isn't served
    from "/") need an absolute URL — TTS_PREVIEW_PUBLIC_BASE_URL prefixes
    it even in local storage mode."""
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_STORAGE", "local")
    monkeypatch.setattr(preview_storage, "LOCAL_PREVIEW_DIR", str(tmp_path))
    monkeypatch.setattr(
        preview_storage, "TTS_PREVIEW_PUBLIC_BASE_URL", "http://localhost:8931"
    )
    url = await preview_storage.store_preview(
        "cartesia", "v1", "en", "k123", b"RIFFdata"
    )
    assert url == "http://localhost:8931/tts-previews/cartesia/v1/en-k123.wav"
    assert (tmp_path / "cartesia" / "v1" / "en-k123.wav").read_bytes() == b"RIFFdata"


@pytest.mark.asyncio
async def test_gcs_store_returns_public_url(monkeypatch):
    """GCS mode uses `get_gcs_bucket` (not the recordings-only `GCSStorage`
    class, which hardcodes the global `GCS_BUCKET`) so the dedicated
    `TTS_PREVIEW_GCS_BUCKET` can be targeted. No real GCS call is made."""
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_STORAGE", "gcs")
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_GCS_BUCKET", "my-preview-bucket")
    monkeypatch.setattr(
        preview_storage,
        "TTS_PREVIEW_PUBLIC_BASE_URL",
        "https://storage.googleapis.com/my-preview-bucket",
    )

    calls: dict = {}

    class _StubBlob:
        def upload_from_string(self, data, content_type=None):
            calls["data"] = data
            calls["content_type"] = content_type

    class _StubBucket:
        def blob(self, path):
            calls["path"] = path
            return _StubBlob()

    def _stub_get_gcs_bucket(bucket_name):
        calls["bucket_name"] = bucket_name
        return _StubBucket()

    monkeypatch.setattr(preview_storage, "get_gcs_bucket", _stub_get_gcs_bucket)

    url = await preview_storage.store_preview(
        "elevenlabs", "v9", "hi", "abc", b"RIFFdata"
    )

    assert (
        url
        == "https://storage.googleapis.com/my-preview-bucket/tts-previews/elevenlabs/v9/hi-abc.wav"
    )
    assert calls["bucket_name"] == "my-preview-bucket"
    assert calls["path"] == "tts-previews/elevenlabs/v9/hi-abc.wav"
    assert calls["data"] == b"RIFFdata"
    assert calls["content_type"] == "audio/wav"


@pytest.mark.asyncio
async def test_path_traversal_component_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_STORAGE", "local")
    monkeypatch.setattr(preview_storage, "LOCAL_PREVIEW_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        await preview_storage.store_preview("cartesia", "../evil", "en", "k", b"x")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_local_store_leaves_no_temp_files(tmp_path, monkeypatch):
    """Writes are temp-file + os.replace — nothing partial in the served dir."""
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_STORAGE", "local")
    monkeypatch.setattr(preview_storage, "LOCAL_PREVIEW_DIR", str(tmp_path))
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_PUBLIC_BASE_URL", "")
    await preview_storage.store_preview("cartesia", "v1", "en", "k123", b"RIFFdata")
    leftovers = [p for p in tmp_path.rglob("*") if p.suffix == ".tmp"]
    assert leftovers == []


@pytest.fixture()
def local_manifest_env(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_STORAGE", "local")
    monkeypatch.setattr(preview_storage, "LOCAL_PREVIEW_DIR", str(tmp_path))
    monkeypatch.setattr(preview_storage, "TTS_PREVIEW_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(preview_storage, "_manifest_cache", None)
    return tmp_path


@pytest.mark.asyncio
async def test_manifest_missing_reads_as_empty(local_manifest_env):
    assert await preview_storage.load_manifest() == {}


@pytest.mark.asyncio
async def test_update_previews_round_trips_through_manifest(local_manifest_env):
    entries = [
        {"language": "en", "url": "/tts-previews/x.wav", "content_key": "k1"},
    ]
    await preview_storage.update_previews("cartesia", "v1", entries)
    manifest = await preview_storage.load_manifest()
    assert manifest == {"cartesia/v1": entries}
    # persisted, not just cached: a fresh read from disk sees it too
    assert (await preview_storage.load_manifest_fresh()) == {"cartesia/v1": entries}


@pytest.mark.asyncio
async def test_update_previews_composes_across_voices(local_manifest_env):
    """Sequential updates read-modify-write fresh — the second voice must not
    clobber the first."""
    await preview_storage.update_previews("cartesia", "v1", [{"language": "en"}])
    await preview_storage.update_previews("sarvam", "shreya", [{"language": "hi"}])
    manifest = await preview_storage.load_manifest_fresh()
    assert set(manifest) == {"cartesia/v1", "sarvam/shreya"}


@pytest.mark.asyncio
async def test_load_manifest_fails_soft_but_update_fails_hard(
    local_manifest_env, monkeypatch
):
    """GET path degrades to {} on storage errors; the reconcile write path
    must abort instead of overwriting good state with a partial manifest."""

    async def boom():
        raise RuntimeError("storage down")

    monkeypatch.setattr(preview_storage, "load_manifest_fresh", boom)
    assert await preview_storage.load_manifest() == {}
    with pytest.raises(RuntimeError):
        await preview_storage.update_previews("cartesia", "v1", [])


@pytest.mark.asyncio
async def test_corrupt_manifest_self_heals(local_manifest_env):
    """A manifest that exists but doesn't parse loads as {} (logged) instead
    of raising — so reconcile can regenerate and its successful write repairs
    the file, rather than every future write failing forever."""
    (local_manifest_env / "manifest.json").write_text("{not json", encoding="utf-8")
    assert await preview_storage.load_manifest_fresh() == {}

    await preview_storage.update_previews("cartesia", "v1", [{"language": "en"}])
    assert await preview_storage.load_manifest_fresh() == {
        "cartesia/v1": [{"language": "en"}]
    }
