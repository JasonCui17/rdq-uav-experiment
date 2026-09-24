"""Portable runtime-path resolution for repository tools.

Source/config paths are repository-relative by default. Large external assets
can be relocated without editing tracked YAML by setting the documented
``RDQ_*`` environment variables.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Environment variable -> nested effective-config field.
RUNTIME_PATH_OVERRIDES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("RDQ_DATA_ROOT", ("data", "root")),
    ("RDQ_SPLIT_FILE", ("data", "split_file")),
    ("RDQ_ANNOTATION_MANIFEST", ("data", "annotation_manifest")),
    ("RDQ_CAMERA_CONFIG", ("data", "camera_config")),
    ("RDQ_GEOMETRY_CALIBRATION", ("data", "geometry_calibration")),
    ("RDQ_LIDAR_CONFIG", ("initialization", "lidar_config")),
    ("RDQ_LIDAR_CHECKPOINT", ("initialization", "lidar_checkpoint")),
    ("RDQ_DETREX_ROOT", ("initialization", "dino_root")),
    ("RDQ_DINO_CONFIG", ("initialization", "dino_config")),
    ("RDQ_DINO_CHECKPOINT", ("initialization", "dino_checkpoint")),
    ("RDQ_P6_CONFIG", ("initialization", "p6_config")),
    ("RDQ_OUTPUT_DIR", ("experiment", "output_dir")),
)


def resolve_project_path(value: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> Path:
    """Resolve one path after expanding ``~`` and environment variables."""

    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    if re.search(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
        raise ValueError(f"path contains an unresolved environment variable: {value!s}")
    path = Path(expanded)
    return path if path.is_absolute() else Path(root) / path


def _set_nested(config: MutableMapping[str, Any], keys: tuple[str, ...], value: str) -> None:
    node: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, MutableMapping):
            raise KeyError(f"runtime path override requires config field {'.'.join(keys)}")
        node = child
    if keys[-1] not in node:
        raise KeyError(f"runtime path override requires config field {'.'.join(keys)}")
    node[keys[-1]] = value


def apply_runtime_path_overrides(
    config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a copied config with explicit non-empty ``RDQ_*`` overrides."""

    effective = copy.deepcopy(dict(config))
    environment = os.environ if environ is None else environ
    for variable, keys in RUNTIME_PATH_OVERRIDES:
        value = environment.get(variable)
        if value:
            _set_nested(effective, keys, value)
    return effective


def ensure_detrex_config_link(detrex_root: Path) -> Path:
    """Ensure detrex's source checkout exposes packaged configs portably.

    Detrex's setup helper creates this link with an absolute source path. That
    link breaks when a checkout is copied to another server. A real copied
    config directory is accepted; a symlink is normalized to a relative link.
    """

    root = Path(detrex_root)
    links = (
        (root / "configs", root / "detrex/config/configs"),
        (
            root / "detectron2/configs",
            root / "detectron2/detectron2/model_zoo/configs",
        ),
    )
    for source, destination in links:
        if not source.is_dir():
            raise FileNotFoundError(f"vendored config source is missing: {source}")
        if destination.is_dir() and not destination.is_symlink():
            continue
        relative_target = Path(os.path.relpath(source, destination.parent))
        if destination.is_symlink():
            if Path(os.readlink(destination)) == relative_target:
                continue
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(f"unexpected vendored config path: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(relative_target, target_is_directory=True)
    return links[0][1]
