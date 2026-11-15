"""Shared test setup for the smoke tests.

The training scripts were written against the legacy ``gym`` API (``reset()``
returns only the observation, ``step()`` returns four values, ``env.seed()``
exists). The repository installs ``gymnasium`` instead, so these tests register
a tiny ``gym`` stand-in that wraps gymnasium with the legacy call signatures.
The scripts themselves are not modified.
"""
import importlib.util
import sys
import types
from pathlib import Path

import gymnasium
import pytest

ROOT = Path(__file__).resolve().parent.parent


class _LegacyEnv:
    """Gymnasium environment exposing the legacy gym call signatures."""

    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = 500

    def seed(self, seed=None):
        self._env.reset(seed=seed)

    def reset(self):
        obs, _info = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated or truncated, info

    def close(self):
        self._env.close()


@pytest.fixture(autouse=True)
def legacy_gym(monkeypatch):
    shim = types.ModuleType("gym")
    shim.make = lambda name, **kw: _LegacyEnv(gymnasium.make(name, **kw))
    monkeypatch.setitem(sys.modules, "gym", shim)


@pytest.fixture
def load_script(monkeypatch, tmp_path):
    """Import a training script by path, without running its training loop.

    The script's folder is put on ``sys.path`` for the duration of the import
    (several folders ship their own ``PER.py``), and the working directory is a
    temporary folder so no model files are written into the repository.
    """
    monkeypatch.chdir(tmp_path)

    def _load(folder, filename):
        script_dir = ROOT / folder
        monkeypatch.syspath_prepend(str(script_dir))
        sys.modules.pop("PER", None)
        spec = importlib.util.spec_from_file_location(
            f"smoke_{filename[:-3]}", script_dir / filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules.pop("PER", None)
        return module

    return _load
