import logging
import glob
import os
import tempfile
from typing import Any, Dict, Optional

import ray

logger = logging.getLogger(__name__)


def cleanup_stale_runtime_resources(temp_dir: Optional[str] = None) -> int:
    """Remove stale Ray runtime resource bundles from previous sessions.

    Parameters
    ----------
    temp_dir : str, optional
        Ray temp directory root. If None, defaults to ``/tmp/ray``.

    Returns
    -------
    int
        Number of stale runtime resource directories removed.
    """
    root_dir = os.path.abspath(os.path.expanduser(temp_dir)) if temp_dir else "/tmp/ray"
    patterns = [
        os.path.join(root_dir, "session_*", "runtime_resources", "py_modules_files"),
        os.path.join(root_dir, "session_*", "runtime_resources", "working_dir_files"),
    ]

    removed = 0
    for pattern in patterns:
        for stale_path in glob.glob(pattern):
            try:
                import shutil

                shutil.rmtree(stale_path, ignore_errors=True)
                removed += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to remove stale Ray runtime cache '%s': %s", stale_path, exc)

    return removed


def ensure_writable_directory(path: str, label: str) -> str:
    """Create a directory if needed and verify it is writable."""
    if path is None:
        raise ValueError(f"{label} cannot be None")

    resolved = os.path.abspath(os.path.expanduser(path))
    os.makedirs(resolved, exist_ok=True)

    probe_fd, probe_path = tempfile.mkstemp(prefix="llamas_ray_", dir=resolved)
    os.close(probe_fd)
    os.remove(probe_path)

    return resolved


def init_ray_cluster(
    *,
    num_cpus: Optional[int] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    object_store_memory: Optional[int] = None,
    ignore_reinit_error: bool = False,
    ray_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize Ray after shutdown with optional temp/spill path controls.

    Parameters
    ----------
    num_cpus : int, optional
        Number of CPUs for Ray.
    runtime_env : dict, optional
        Runtime environment passed to ray.init.
    object_store_memory : int, optional
        Bytes for Ray object store memory.
    ignore_reinit_error : bool
        Forwarded to ray.init for compatibility with existing behavior.
    ray_options : dict, optional
        Optional pipeline-level Ray options dictionary supporting:
        - ray_temp_dir
        - ray_object_spilling_directory

    Returns
    -------
    dict
        Effective keyword arguments used for ray.init.
    """
    effective: Dict[str, Any] = {}

    if num_cpus is not None:
        effective["num_cpus"] = int(num_cpus)
    if runtime_env is not None:
        effective["runtime_env"] = runtime_env
    if object_store_memory is not None:
        effective["object_store_memory"] = int(object_store_memory)

    effective["ignore_reinit_error"] = bool(ignore_reinit_error)

    if ray_options:
        temp_dir = ray_options.get("ray_temp_dir")
        spill_dir = ray_options.get("ray_object_spilling_directory")

        if temp_dir:
            effective["_temp_dir"] = ensure_writable_directory(temp_dir, "ray_temp_dir")
        if spill_dir:
            effective["object_spilling_directory"] = ensure_writable_directory(
                spill_dir,
                "ray_object_spilling_directory",
            )

    ray.shutdown()

    cleanup_root = effective.get("_temp_dir")
    removed = cleanup_stale_runtime_resources(cleanup_root)
    if removed:
        logger.info(
            "Cleaned %d stale Ray runtime resource directory(s) under %s",
            removed,
            cleanup_root if cleanup_root else "/tmp/ray",
        )

    try:
        ray.init(**effective)
    except TypeError as exc:
        # Older Ray builds may not support object_spilling_directory.
        if "object_spilling_directory" in effective:
            logger.warning(
                "ray.init does not accept object_spilling_directory in this Ray version; "
                "retrying without it. Error: %s",
                exc,
            )
            effective.pop("object_spilling_directory", None)
            ray.init(**effective)
        else:
            raise

    return effective
