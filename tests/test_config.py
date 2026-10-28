"""Unit tests for ``src.config``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import config
from src.data_prep import PATHOLOGIES as DATA_PREP_PATHOLOGIES


def test_pathologies_match_data_prep() -> None:
    assert config.PATHOLOGIES == DATA_PREP_PATHOLOGIES
    assert config.NUM_CLASSES == 14


def test_priority_map_covers_every_class_with_valid_levels() -> None:
    assert set(config.PRIORITY_MAP) == set(config.PATHOLOGIES)
    assert set(config.PRIORITY_MAP.values()) <= set(config.PRIORITY_LEVELS)
    assert set(config.PRIORITY_COLORS) == set(config.PRIORITY_LEVELS)
    assert config.PRIORITY_MAP["Pneumothorax"] == "IMMEDIATE"


def test_locate_processed_dir_uses_environment_variable(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    np.save(tmp_path / config.array_name(), np.zeros((1, 2, 2), dtype=np.uint8))
    monkeypatch.setenv(config.ENV_DATA_DIR, str(tmp_path))
    assert config.locate_processed_dir() == tmp_path


def test_locate_processed_dir_rejects_wrong_environment_path(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_DATA_DIR, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        config.locate_processed_dir()


def test_runs_dir_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_RUNS_DIR, str(tmp_path))
    assert config.runs_dir() == tmp_path


def test_train_config_to_dict_is_json_friendly() -> None:
    import json

    settings = config.TrainConfig(seed=43, pos_weight_mode="capped").to_dict()
    assert settings["seed"] == 43 and settings["augment"]["horizontal_flip"] is False
    json.dumps(settings)  # must not raise