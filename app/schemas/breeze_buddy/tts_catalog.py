from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

CATALOG_PROVIDERS: tuple[str, ...] = (
    "elevenlabs",
    "cartesia",
    "sarvam",
    "gemini",
    "google",
    "soniox",
)


class CatalogVoiceEntry(BaseModel):
    """One curated voice as authored in app/ai/voice/tts/catalog.json.

    Internal representation — previews are not part of the entry; they are
    tracked in the preview-storage manifest and merged in at read time.
    """

    provider: str
    voice_id: str
    display_name: str
    models: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    enabled: bool = True
    residency: Optional[str] = None
    style_params: Optional[dict] = None

    @field_validator("provider")
    @classmethod
    def _provider_known(cls, v: str) -> str:
        if v not in CATALOG_PROVIDERS:
            raise ValueError(
                f"Unknown provider {v!r}; must be one of {CATALOG_PROVIDERS}"
            )
        return v


class VoicePreview(BaseModel):
    language: str
    url: str


class CatalogVoice(BaseModel):
    provider: str
    voice_id: str
    display_name: str
    models: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_default: bool = False
    previews: List[VoicePreview] = Field(default_factory=list)


class VoicesResponse(BaseModel):
    voices: List[CatalogVoice]
