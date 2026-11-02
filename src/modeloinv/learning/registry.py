"""Model Version Registry and Checkpoint Lifecycle Management for modeloINV.

Provides:
- ModelMetadata: Dataclass capturing semantic version, timestamps, training metrics,
  validation metrics, artifact paths, and champion status.
- ModelRegistry: Central registry managing atomic persistence to models/registry.json,
  semantic versioning (vMAJOR.MINOR.PATCH), artifact serialization (.joblib, .pt, config.json),
  candidate evaluation promotion, and rollback capabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import json
import os
import re
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import torch

from modeloinv.strategy.ml_model import HistGradientBoostingModel
from modeloinv.strategy.rl_agent import PolicyAgent


@dataclass
class ModelMetadata:
    """Complete metadata and evaluation metrics for a specific model version."""
    version: str
    timestamp: datetime.datetime
    trigger_reason: str
    training_window: Dict[str, Any]
    ml_metrics: Dict[str, float] = field(default_factory=dict)
    rl_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    status: str = "CANDIDATE"  # "CHAMPION", "CANDIDATE", "ARCHIVED", "REJECTED"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata to dictionary representation."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "trigger_reason": self.trigger_reason,
            "training_window": self.training_window,
            "ml_metrics": self.ml_metrics,
            "rl_metrics": self.rl_metrics,
            "validation_metrics": self.validation_metrics,
            "artifacts": self.artifacts,
            "status": self.status,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Constructs ModelMetadata from dictionary."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

        return cls(
            version=str(data["version"]),
            timestamp=ts,
            trigger_reason=str(data.get("trigger_reason", "manual")),
            training_window=dict(data.get("training_window", {})),
            ml_metrics=dict(data.get("ml_metrics", {})),
            rl_metrics=dict(data.get("rl_metrics", {})),
            validation_metrics=dict(data.get("validation_metrics", {})),
            artifacts=dict(data.get("artifacts", {})),
            status=str(data.get("status", "CANDIDATE")),
            description=str(data.get("description", "")),
        )


class ModelRegistry:
    """Atomic model version registry managing persistence, checkpoints, and lifecycle."""

    SEMVER_REGEX = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

    def __init__(
        self,
        registry_path: str = "models/registry.json",
        models_dir: str = "models",
    ) -> None:
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.active_champion_version: Optional[str] = None
        self.models: List[ModelMetadata] = []
        self._lock = threading.RLock()
        self._load_registry()

    def _load_registry(self) -> None:
        """Loads registry from disk if it exists."""
        if not os.path.exists(self.registry_path):
            return

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_champion_version = data.get("active_champion_version")
            self.models = [ModelMetadata.from_dict(m) for m in data.get("models", [])]
        except Exception:
            # Corrupted or unreadable, start fresh
            self.active_champion_version = None
            self.models = []

    def _save_registry_atomic(self) -> None:
        """Persists registry state atomically via temporary file and replace."""
        with self._lock:
            dir_name = os.path.dirname(self.registry_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            payload = {
                "active_champion_version": self.active_champion_version,
                "last_updated": datetime.datetime.utcnow().isoformat() + "Z",
                "models": [m.to_dict() for m in self.models],
            }

            # Atomic write to temporary file in same directory
            temp_fd, temp_file_path = tempfile.mkstemp(
                prefix="registry_",
                suffix=".tmp",
                dir=dir_name if dir_name else None,
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic replace with retry for Windows NTFS file lock survival
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        os.replace(temp_file_path, self.registry_path)
                        break
                    except PermissionError:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(0.05 * (2 ** attempt))
            except Exception:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
                raise

    @classmethod
    def parse_semver(cls, version_str: str) -> Tuple[int, int, int]:
        """Parses semantic version string vMAJOR.MINOR.PATCH into tuple (major, minor, patch)."""
        match = cls.SEMVER_REGEX.match(version_str.strip())
        if not match:
            raise ValueError(f"Invalid semantic version string: '{version_str}'. Must follow vMAJOR.MINOR.PATCH.")
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    def get_latest_version(self) -> Optional[str]:
        """Returns the highest semantic version registered, or None if registry is empty."""
        if not self.models:
            return None

        def version_key(m: ModelMetadata) -> Tuple[int, int, int]:
            try:
                return self.parse_semver(m.version)
            except ValueError:
                return (0, 0, 0)

        highest = max(self.models, key=version_key)
        return highest.version

    def next_version(self, bump_type: str = "patch") -> str:
        """Determines the next version string based on bump type ('major', 'minor', 'patch')."""
        if bump_type not in ("major", "minor", "patch"):
            raise ValueError(f"Unknown bump_type: '{bump_type}'. Choose 'major', 'minor', or 'patch'.")
        with self._lock:
            latest = self.get_latest_version()
            if not latest:
                return "v1.0.0"

            major, minor, patch = self.parse_semver(latest)
            if bump_type == "major":
                return f"v{major + 1}.0.0"
            elif bump_type == "minor":
                return f"v{major}.{minor + 1}.0"
            elif bump_type == "patch":
                return f"v{major}.{minor}.{patch + 1}"
            else:
                raise ValueError(f"Unknown bump_type: '{bump_type}'. Choose 'major', 'minor', or 'patch'.")

    def register_candidate(
        self,
        metadata: ModelMetadata,
        ml_model: Optional[HistGradientBoostingModel] = None,
        rl_agent: Optional[PolicyAgent] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelMetadata:
        """Saves model artifacts to models/{version}/ and adds candidate to registry."""
        with self._lock:
            ver = metadata.version
            version_dir = os.path.join(self.models_dir, ver)
            os.makedirs(version_dir, exist_ok=True)

            # Save ML model artifact
            if ml_model is not None:
                ml_path = os.path.join(version_dir, "ml_model.joblib")
                ml_model.save(ml_path)
                metadata.artifacts["ml_model_path"] = ml_path

            # Save RL policy agent artifact
            if rl_agent is not None:
                rl_path = os.path.join(version_dir, "policy.pt")
                rl_agent.save(rl_path)
                metadata.artifacts["rl_policy_path"] = rl_path

            # Save config snapshot
            if config is not None:
                config_path = os.path.join(version_dir, "config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                metadata.artifacts["config_path"] = config_path

            # Check if version already exists
            for idx, existing in enumerate(self.models):
                if existing.version == ver:
                    self.models[idx] = metadata
                    self._save_registry_atomic()
                    return metadata

            self.models.append(metadata)
            self._save_registry_atomic()
            return metadata

    def promote_to_champion(self, version: str) -> bool:
        """Promotes candidate version to active CHAMPION and archives previous CHAMPION."""
        with self._lock:
            target_model = self.get_model(version)
            if target_model is None:
                return False

            # Archive current champion
            if self.active_champion_version and self.active_champion_version != version:
                current_champ = self.get_model(self.active_champion_version)
                if current_champ:
                    current_champ.status = "ARCHIVED"

            target_model.status = "CHAMPION"
            self.active_champion_version = version
            self._save_registry_atomic()
            return True

    def reject_candidate(self, version: str, reason: str = "") -> bool:
        """Marks candidate version as REJECTED with rejection reason recorded."""
        with self._lock:
            target_model = self.get_model(version)
            if target_model is None:
                return False

            target_model.status = "REJECTED"
            if reason:
                target_model.description = f"Rejected: {reason}"
            self._save_registry_atomic()
            return True

    def rollback(self, target_version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Rolls back to specified version or most recent ARCHIVED champion.

        Returns:
            Promoted ModelMetadata or None if rollback is not possible.
        """
        with self._lock:
            if target_version is not None:
                target = self.get_model(target_version)
                if target is None:
                    return None
            else:
                # Look for most recent ARCHIVED model
                archived_models = [m for m in reversed(self.models) if m.status == "ARCHIVED"]
                if not archived_models:
                    return None
                target = archived_models[0]

            # Archive current champion
            if self.active_champion_version:
                current_champ = self.get_model(self.active_champion_version)
                if current_champ and current_champ.version != target.version:
                    current_champ.status = "ARCHIVED"

            target.status = "CHAMPION"
            self.active_champion_version = target.version
            self._save_registry_atomic()
            return target

    def get_active_champion(self) -> Optional[ModelMetadata]:
        """Retrieves active champion metadata."""
        if not self.active_champion_version:
            # Fall back to finding any model with CHAMPION status
            for m in reversed(self.models):
                if m.status == "CHAMPION":
                    self.active_champion_version = m.version
                    return m
            return None
        return self.get_model(self.active_champion_version)

    def get_model(self, version: str) -> Optional[ModelMetadata]:
        """Retrieves metadata by version string."""
        for m in self.models:
            if m.version == version:
                return m
        return None

    def list_models(self) -> List[ModelMetadata]:
        """Returns list of all registered models."""
        return list(self.models)

    def load_artifacts(
        self,
        version: str,
    ) -> Tuple[Optional[HistGradientBoostingModel], Optional[PolicyAgent], Optional[Dict[str, Any]]]:
        """Loads serialized artifacts for a given version.

        Returns:
            Tuple of (ml_model, rl_agent, config_dict)
        """
        meta = self.get_model(version)
        if meta is None:
            raise ValueError(f"Version '{version}' not found in registry.")

        ml_model = None
        if "ml_model_path" in meta.artifacts and os.path.exists(meta.artifacts["ml_model_path"]):
            ml_model = HistGradientBoostingModel.load(meta.artifacts["ml_model_path"])

        rl_agent = None
        if "rl_policy_path" in meta.artifacts and os.path.exists(meta.artifacts["rl_policy_path"]):
            rl_agent = PolicyAgent()
            rl_agent.load(meta.artifacts["rl_policy_path"])

        config = None
        if "config_path" in meta.artifacts and os.path.exists(meta.artifacts["config_path"]):
            with open(meta.artifacts["config_path"], "r", encoding="utf-8") as f:
                config = json.load(f)

        return ml_model, rl_agent, config
