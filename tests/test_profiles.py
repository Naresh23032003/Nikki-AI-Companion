"""Tests for app/profiles.py, including the N10 privacy fix: a profile's
phone number can be overridden via COMPANION_NUMBER_<ID> in .env instead of
living in the tracked config.yaml.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.profiles import load_profiles, normalize_number


def _settings(profiles):
    return SimpleNamespace(
        profiles=profiles,
        persona_active="luna",
        wa_session_id="main",
        db_path=Path("companion.db"),
        memory_collection="companion_memories",
    )


class TestNormalizeNumber:
    def test_strips_plus_and_dashes(self):
        assert normalize_number("+91 55512-30001") == "915551230001"

    def test_strips_whatsapp_jid_suffix(self):
        assert normalize_number("919876543210@c.us") == "919876543210"

    def test_empty_input(self):
        assert normalize_number("") == ""
        assert normalize_number(None) == ""


class TestEnvNumberOverride:
    def test_env_override_wins_over_yaml_placeholder(self, monkeypatch):
        monkeypatch.setenv("COMPANION_NUMBER_MAIN", "915551230001")
        settings = _settings([
            {"id": "main", "number": "919876543210", "persona": "luna", "default": True},
        ])
        registry = load_profiles(settings)
        assert registry.default.number == "915551230001"

    def test_no_env_var_falls_back_to_yaml(self, monkeypatch):
        monkeypatch.delenv("COMPANION_NUMBER_MAIN", raising=False)
        settings = _settings([
            {"id": "main", "number": "919876543210", "persona": "luna", "default": True},
        ])
        registry = load_profiles(settings)
        assert registry.default.number == "919876543210"

    def test_override_is_per_profile_id(self, monkeypatch):
        monkeypatch.setenv("COMPANION_NUMBER_MAIN", "915551230001")
        monkeypatch.setenv("COMPANION_NUMBER_FRIEND", "915551230002")
        settings = _settings([
            {"id": "main", "number": "919876543210", "persona": "luna", "default": True},
            {"id": "friend", "number": "919876543211", "persona": "aria"},
        ])
        registry = load_profiles(settings)
        by_id = {p.id: p.number for p in registry}
        assert by_id == {"main": "915551230001", "friend": "915551230002"}

    def test_env_override_is_normalized_too(self, monkeypatch):
        """A messily-formatted override must still normalize, matching the
        YAML path's behaviour, or bridge/WhatsApp number matching breaks."""
        monkeypatch.setenv("COMPANION_NUMBER_MAIN", "+91 55512-30001")
        settings = _settings([
            {"id": "main", "number": "919876543210", "persona": "luna", "default": True},
        ])
        registry = load_profiles(settings)
        assert registry.default.number == "915551230001"


class TestLoadProfilesBasics:
    def test_no_profiles_block_falls_back_to_legacy_single_profile(self):
        settings = _settings(None)
        registry = load_profiles(settings)
        assert len(registry) == 1
        assert registry.default.is_default

    def test_malformed_entry_is_skipped_not_fatal(self, monkeypatch):
        monkeypatch.delenv("COMPANION_NUMBER_MAIN", raising=False)
        settings = _settings([
            "not-a-dict",
            {"id": "main", "number": "919876543210", "persona": "luna", "default": True},
        ])
        registry = load_profiles(settings)
        assert len(registry) == 1

    def test_duplicate_number_is_skipped(self, monkeypatch):
        monkeypatch.delenv("COMPANION_NUMBER_MAIN", raising=False)
        monkeypatch.delenv("COMPANION_NUMBER_FRIEND", raising=False)
        settings = _settings([
            {"id": "main", "number": "919876543210", "persona": "luna", "default": True},
            {"id": "friend", "number": "919876543210", "persona": "aria"},
        ])
        registry = load_profiles(settings)
        assert len(registry) == 1
