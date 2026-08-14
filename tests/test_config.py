from __future__ import annotations

from pathlib import Path

import pytest

from llama_server_pool.config import Settings


def test_ui_environment_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("LLAMA_POOL_UI_ENABLED", "off")
    monkeypatch.setenv("LLAMA_POOL_MODEL_DISCOVERY_ROOT", str(root))

    settings = Settings.from_env()

    assert settings.ui_enabled is False
    assert settings.model_discovery_root == str(root.resolve())


def test_invalid_ui_boolean_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_POOL_UI_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        Settings.from_env()


def test_missing_discovery_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be accessed"):
        Settings(model_discovery_root=str(tmp_path / "missing"))


def test_profiles_file_defaults_to_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("LLAMA_POOL_PROFILES_FILE", raising=False)

    settings = Settings.from_env()

    assert settings.profiles_file == str(
        (tmp_path / "llama-server-pool" / "profiles.json").resolve()
    )


def test_profiles_file_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "shared" / "models.json"
    monkeypatch.setenv("LLAMA_POOL_PROFILES_FILE", str(path))

    assert Settings.from_env().profiles_file == str(path.resolve())
