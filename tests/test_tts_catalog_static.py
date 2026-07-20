# tests/test_tts_catalog_static.py — invariants of the checked-in catalog.
from app.ai.voice.tts.catalog import get_all_voices, get_enabled_voices
from app.core.config.dynamic import BB_SPEECH_PROVIDER_DEFAULTS
from app.schemas.breeze_buddy.tts_catalog import CATALOG_PROVIDERS


def test_catalog_parses_and_validates():
    voices = get_all_voices()
    assert voices, "catalog.json must not be empty"
    assert all(v.provider in CATALOG_PROVIDERS for v in voices)


def test_catalog_pairs_unique():
    pairs = [(v.provider, v.voice_id) for v in get_all_voices()]
    assert len(pairs) == len(set(pairs))


def test_enabled_subset():
    assert {(v.provider, v.voice_id) for v in get_enabled_voices()} <= {
        (v.provider, v.voice_id) for v in get_all_voices()
    }


def test_catalog_covers_static_provider_defaults():
    """Every hardcoded default voice must exist in the catalog — otherwise the
    picker can't show the voice a template actually falls back to. (The old
    seed script guaranteed this at seed time; now it's a code invariant.)"""
    catalog_pairs = {(v.provider, v.voice_id) for v in get_enabled_voices()}
    for provider in CATALOG_PROVIDERS:
        default_voice = BB_SPEECH_PROVIDER_DEFAULTS.get(provider, {}).get("voice_id")
        if default_voice:
            assert (
                provider,
                default_voice,
            ) in catalog_pairs, (
                f"static default {provider}/{default_voice} missing from catalog.json"
            )
    # gemini has no BB_SPEECH_PROVIDER_DEFAULTS entry; the handler falls back
    # to "Kore", which must therefore exist too.
    assert ("gemini", "Kore") in catalog_pairs
