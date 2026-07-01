"""COLMAP reconstruction + dataset import panel."""

from __future__ import annotations

import ctypes
import gc
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

import lichtfeld as lf
import pycolmap
from pycolmap import CameraMode

try:
    import numpy as np
except ImportError:
    np = None

try:
    from lfs_plugins import ScrubFieldController, ScrubFieldSpec
except ImportError:
    from lfs_plugins.scrub_fields import ScrubFieldController, ScrubFieldSpec


class ReconStage(Enum):
    IDLE = "idle"
    CHECKING = "checking"
    FEATURE_EXTRACTION = "feature_extraction"
    MATCHING = "matching"
    MAPPING = "mapping"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


SCRUB_FIELD_SPECS = {
    "sift_max_num_features": ScrubFieldSpec(
        min_value=1024.0,
        max_value=10000.0,
        step=512.0,
        fmt="%d",
        data_type=int,
    ),
    "sift_max_num_matches": ScrubFieldSpec(
        min_value=1024.0,
        max_value=10000.0,
        step=512.0,
        fmt="%d",
        data_type=int,
    ),
    "exhaustive_block_size": ScrubFieldSpec(
        min_value=5.0,
        max_value=50.0,
        step=1.0,
        fmt="%d",
        data_type=int,
    ),
    "ba_global_max_num_iterations": ScrubFieldSpec(
        min_value=10.0,
        max_value=100.0,
        step=5.0,
        fmt="%d",
        data_type=int,
    ),
}

PRESET_LOW_PARAMS = dict(
    camera_model="OPENCV",
    single_camera=True,
    downsample_multiplier=2,
    sift_max_num_features=1536,
    matcher="sequential",
    reconstruction_mode="incremental",
    use_view_graph_calibration=False,
    sift_max_num_matches=1024,
    exhaustive_block_size=10,
    ba_global_max_num_iterations=20,
)

PRESET_NORMAL_PARAMS = dict(
    camera_model="OPENCV",
    single_camera=True,
    downsample_multiplier=1,
    sift_max_num_features=2048,
    matcher="exhaustive",
    reconstruction_mode="incremental",
    use_view_graph_calibration=False,
    sift_max_num_matches=2048,
    exhaustive_block_size=15,
    ba_global_max_num_iterations=50,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
COLMAP_MODEL_BASENAMES = ("cameras", "images", "points3D")


def _try_set_attr(obj, attr: str, value) -> bool:
    """Best-effort set for pybind option objects (older pycolmap builds may lack some fields)."""
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def _trim_process_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _stage_flat_directory(
    source_dir: Path,
    target_dir: Path,
    *,
    allowed_extensions: Optional[tuple[str, ...]] = None,
) -> int:
    if _same_path(source_dir, target_dir):
        return sum(
            1
            for entry in source_dir.iterdir()
            if entry.is_file()
            and (
                allowed_extensions is None or entry.suffix.lower() in allowed_extensions
            )
        )

    _reset_directory(target_dir)

    staged_count = 0
    for entry in sorted(source_dir.iterdir(), key=lambda path: path.name.lower()):
        if not entry.is_file():
            continue
        if (
            allowed_extensions is not None
            and entry.suffix.lower() not in allowed_extensions
        ):
            continue
        _link_or_copy_file(entry, target_dir / entry.name)
        staged_count += 1

    return staged_count


def _stage_and_downsample_directory(
    source_dir: Path,
    target_dir: Path,
    *,
    downsample_multiplier: int = 1,
    allowed_extensions: Optional[tuple[str, ...]] = None,
    warn_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """Stage images with downsampling. downsample_multiplier of 1 means no downsampling (1x).
    2 means 0.5x scale (half resolution), 4 means 0.25x, 8 means 0.125x."""
    if downsample_multiplier <= 1:
        # No downsampling needed, use regular staging
        return _stage_flat_directory(
            source_dir, target_dir, allowed_extensions=allowed_extensions
        )

    if _same_path(source_dir, target_dir):
        return sum(
            1
            for entry in source_dir.iterdir()
            if entry.is_file()
            and (
                allowed_extensions is None or entry.suffix.lower() in allowed_extensions
            )
        )

    _reset_directory(target_dir)

    if Image is None:
        raise RuntimeError(
            "PIL (Pillow) is required for image downsampling but not installed"
        )

    staged_count = 0
    scale = 1.0 / downsample_multiplier

    for entry in sorted(source_dir.iterdir(), key=lambda path: path.name.lower()):
        if not entry.is_file():
            continue
        if (
            allowed_extensions is not None
            and entry.suffix.lower() not in allowed_extensions
        ):
            continue

        target_path = target_dir / entry.name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(entry) as img:
                # Convert to RGB if necessary (for JPEG output)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Calculate new size
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)

                # Ensure minimum size
                new_width = max(new_width, 1)
                new_height = max(new_height, 1)

                # Resize using high-quality downsampling
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save as JPEG with high quality
                resized.save(target_path, "JPEG", quality=95)
                staged_count += 1
        except Exception as e:
            if warn_callback:
                warn_callback(f"[COLMAP] Failed to downsample {entry.name}: {e}")
            # Fall back to copying original
            _link_or_copy_file(entry, target_path)
            staged_count += 1

    return staged_count


def _count_files_in_directory(
    directory: Path,
    *,
    allowed_extensions: Optional[tuple[str, ...]] = None,
) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for entry in directory.iterdir()
        if entry.is_file()
        and (allowed_extensions is None or entry.suffix.lower() in allowed_extensions)
    )


def _is_colmap_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    filenames = {entry.name for entry in path.iterdir() if entry.is_file()}
    return any(
        all(f"{basename}{suffix}" in filenames for basename in COLMAP_MODEL_BASENAMES)
        for suffix in (".bin", ".txt")
    )


def _resolve_sparse_model_path(path: Path) -> Path:
    if _is_colmap_model_directory(path):
        return path
    if not path.is_dir():
        return path
    for child in sorted(path.iterdir(), key=lambda entry: entry.name.lower()):
        if child.is_dir() and _is_colmap_model_directory(child):
            return child
    return path


def _undistort_dataset(
    *,
    sparse_model_path: Path,
    source_images_path: Path,
    output_root_path: Path,
) -> tuple[Path, Path, int]:
    if not hasattr(pycolmap, "undistort_images"):
        raise RuntimeError(
            "This pycolmap build does not expose undistort_images(). "
            "Install a build with image undistortion support."
        )

    _reset_directory(output_root_path)

    undistort_kwargs = {
        "output_path": os.fspath(output_root_path),
        "input_path": os.fspath(sparse_model_path),
        "image_path": os.fspath(source_images_path),
        "output_type": "COLMAP",
    }

    copy_type = getattr(pycolmap, "CopyType", None)
    if copy_type is not None:
        copy_value = getattr(copy_type, "copy", None)
        if copy_value is not None:
            undistort_kwargs["copy_policy"] = copy_value

    if hasattr(pycolmap, "UndistortCameraOptions"):
        undistort_kwargs["undistort_options"] = pycolmap.UndistortCameraOptions()

    try:
        pycolmap.undistort_images(**undistort_kwargs)
    except TypeError as exc:
        fallback_kwargs = dict(undistort_kwargs)
        for key in ("undistort_options", "copy_policy", "output_type"):
            if key not in fallback_kwargs:
                continue
            fallback_kwargs.pop(key, None)
            try:
                pycolmap.undistort_images(**fallback_kwargs)
                break
            except TypeError:
                continue
        else:
            raise RuntimeError(f"pycolmap.undistort_images() failed: {exc}") from exc

    undistorted_images_path = output_root_path / "images"
    undistorted_sparse_path = _resolve_sparse_model_path(output_root_path / "sparse")
    undistorted_image_count = _count_files_in_directory(
        undistorted_images_path,
        allowed_extensions=IMAGE_EXTENSIONS,
    )

    if undistorted_image_count == 0:
        raise RuntimeError(
            f"COLMAP undistortion did not produce any images in {undistorted_images_path}"
        )
    if not _is_colmap_model_directory(undistorted_sparse_path):
        raise RuntimeError(
            f"COLMAP undistortion did not produce a sparse model in {output_root_path / 'sparse'}"
        )

    return undistorted_images_path, undistorted_sparse_path, undistorted_image_count


@dataclass
class ColmapParams:
    # Feature extraction
    camera_model: str = "OPENCV"
    single_camera: bool = True
    downsample_multiplier: int = 1  # 1x, 2x, 4x, 8x
    sift_max_num_features: int = 2048

    # Matching
    matcher: str = "exhaustive"  # exhaustive | sequential
    reconstruction_mode: str = "incremental"  # incremental | global
    use_view_graph_calibration: bool = False
    sift_max_num_matches: int = 2048
    exhaustive_block_size: int = 15

    # Mapping
    ba_global_max_num_iterations: int = 50

    # GPU settings
    use_gpu: bool = True
    gpu_index: int = 0
    gpu_fallback: bool = True


@dataclass
class ReconResult:
    success: bool
    recon_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None
    metrics: Optional["ReconMetrics"] = None


@dataclass(frozen=True)
class ReconMetrics:
    total_points: int = 0
    mean_reprojection_error_px: float = -1.0
    median_reprojection_error_px: float = -1.0
    p90_reprojection_error_px: float = -1.0
    good_ratio: float = -1.0


def _median_sorted(values: list[float]) -> float:
    count = len(values)
    if count == 0:
        return -1.0
    mid = count // 2
    if count % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _percentile_sorted(values: list[float], percentile: float) -> float:
    if not values:
        return -1.0
    if percentile <= 0.0:
        return values[0]
    if percentile >= 100.0:
        return values[-1]

    position = (len(values) - 1) * (percentile / 100.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _compute_recon_metrics(reconstruction) -> ReconMetrics:
    total_points = len(reconstruction.points3D)
    if total_points == 0:
        return ReconMetrics()

    errors: list[float] = []
    for point in reconstruction.points3D.values():
        try:
            error = float(point.error)
        except Exception:
            continue
        if math.isfinite(error):
            errors.append(error)

    if not errors:
        return ReconMetrics(total_points=total_points)

    if np is not None:
        error_array = np.asarray(errors, dtype=float)
        mean_error = float(error_array.mean())
        median_error = float(np.median(error_array))
        p90_error = float(np.percentile(error_array, 90))
    else:
        sorted_errors = sorted(errors)
        mean_error = float(sum(sorted_errors) / len(sorted_errors))
        median_error = float(_median_sorted(sorted_errors))
        p90_error = float(_percentile_sorted(sorted_errors, 90.0))

    good_ratio = float(sum(error < 2.0 for error in errors) / total_points)
    return ReconMetrics(
        total_points=total_points,
        mean_reprojection_error_px=mean_error,
        median_reprojection_error_px=median_error,
        p90_reprojection_error_px=p90_error,
        good_ratio=good_ratio,
    )


def _format_metric_px(value: float) -> str:
    if value < 0.0:
        return "N/A"
    return f"{value:.4f} px"


class ColmapReconJob:
    def __init__(
        self,
        images_dir: str,
        params: ColmapParams,
    ):
        self.images_dir = images_dir
        self.params = params

        self._stage = ReconStage.IDLE
        self._progress = 0.0
        self._status = ""
        self._result: Optional[ReconResult] = None

        self._lock = threading.Lock()
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None
        self._log_lines: deque[str] = deque(maxlen=16)

    @property
    def stage(self) -> ReconStage:
        with self._lock:
            return self._stage

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def result(self) -> Optional[ReconResult]:
        with self._lock:
            return self._result

    @property
    def log_text(self) -> str:
        with self._lock:
            return "\n".join(self._log_lines)

    def is_running(self) -> bool:
        return self.stage in (
            ReconStage.CHECKING,
            ReconStage.FEATURE_EXTRACTION,
            ReconStage.MATCHING,
            ReconStage.MAPPING,
        )

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self._status = "Cancelling..."

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Job already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _update(self, stage: ReconStage, progress: float, status: str):
        with self._lock:
            self._stage = stage
            self._progress = progress
            self._status = status

    def _check_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def _append_log(self, message: str) -> None:
        line = str(message).strip()
        if not line:
            return
        if line.startswith("[COLMAP] "):
            line = line[len("[COLMAP] ") :]
        with self._lock:
            self._log_lines.append(line)

    def _log_info(self, message: str) -> None:
        lf.log.info(message)
        self._append_log(message)

    def _log_warn(self, message: str) -> None:
        lf.log.warn(message)
        self._append_log(message)

    def _log_error(self, message: str) -> None:
        lf.log.error(message)
        self._append_log(message)

    def _ensure_not_cancelled(self):
        if self._check_cancelled():
            raise RuntimeError("Cancelled")

    def _run(self):
        t0 = time.time()
        info = self._log_info
        warn = self._log_warn
        error = self._log_error
        work_root_path: Optional[Path] = None

        try:
            self._update(ReconStage.CHECKING, 2.0, "Checking inputs")

            images_dir = os.path.abspath(self.images_dir)

            if not os.path.isdir(images_dir):
                raise RuntimeError(f"Images directory does not exist: {images_dir}")

            # ------------------------------------------------
            # Count images
            # ------------------------------------------------

            image_files = [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]

            info(f"[COLMAP] Found {len(image_files)} images")

            if len(image_files) < 2:
                raise RuntimeError("Need at least 2 images")

            downsample_multiplier = max(1, int(self.params.downsample_multiplier))
            sift_max_num_features = max(1024, int(self.params.sift_max_num_features))
            sift_max_num_matches = max(1024, int(self.params.sift_max_num_matches))
            exhaustive_block_size = max(5, int(self.params.exhaustive_block_size))
            ba_global_max_num_iterations = max(
                1, int(self.params.ba_global_max_num_iterations)
            )
            matcher_name = self.params.matcher
            reconstruction_mode = self.params.reconstruction_mode
            num_threads = min(8, os.cpu_count() or 4)
            if matcher_name not in ("exhaustive", "sequential"):
                warn(
                    f"[COLMAP] Unknown matcher '{matcher_name}', using exhaustive matching."
                )
                matcher_name = "exhaustive"

            info(
                "[COLMAP] Reconstruction settings: "
                f"matcher={matcher_name}, "
                f"reconstruction_mode={reconstruction_mode}, "
                f"cpu_threads={num_threads}, "
                f"downsample_multiplier={downsample_multiplier}x, "
                f"sift_max_num_features={sift_max_num_features}, "
                f"sift_max_num_matches={sift_max_num_matches}, "
                f"exhaustive_block_size={exhaustive_block_size}, "
                f"ba_global_max_num_iterations={ba_global_max_num_iterations}"
            )

            info("[COLMAP] GPU acceleration enabled")

            # ------------------------------------------------
            # Setup folders
            # ------------------------------------------------

            images_path = Path(images_dir)
            if images_path.name.lower() == "images":
                recon_root_path = images_path.parent
                dataset_images_path = images_path
            else:
                recon_root_path = (
                    images_path.parent / f"{images_path.name}_reconstruction"
                )
                dataset_images_path = recon_root_path / "images"

            recon_root = os.fspath(recon_root_path)
            os.makedirs(recon_root, exist_ok=True)

            # Validate recon_root is writable and disk has space
            if not os.path.isdir(recon_root):
                raise RuntimeError(
                    f"Failed to create reconstruction directory: {recon_root}"
                )

            if not os.access(recon_root, os.W_OK):
                raise RuntimeError(
                    f"Reconstruction directory is not writable: {recon_root}"
                )

            # Test write permissions by attempting to create a temporary file
            test_file = os.path.join(recon_root, ".colmap_write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except OSError as e:
                raise RuntimeError(
                    f"Cannot write to reconstruction directory {recon_root}: {e}. "
                    "Check disk space, permissions, and filesystem health."
                )

            # Check available disk space (warn if < 500 MB)
            try:
                stat = shutil.disk_usage(recon_root)
                free_gb = stat.free / (1024**3)
                if free_gb < 0.5:
                    warn(f"[COLMAP] Low disk space: only {free_gb:.2f} GiB available")
            except Exception:
                pass  # Non-fatal if we can't check disk space

            masks_source_path = images_path.parent / "masks"
            dataset_masks_path = recon_root_path / "masks"
            sparse_root_path = recon_root_path / "sparse"
            work_root_path = recon_root_path / ".colmap"
            staged_images_path = work_root_path / "images"
            mapping_root_path = work_root_path / "mapping"
            temp_publish_root_path = work_root_path / "publish"

            _reset_directory(work_root_path)
            mapping_root_path.mkdir(parents=True, exist_ok=True)

            staged_image_count = _stage_and_downsample_directory(
                images_path,
                staged_images_path,
                downsample_multiplier=downsample_multiplier,
                allowed_extensions=IMAGE_EXTENSIONS,
                warn_callback=warn,
            )

            if masks_source_path.is_dir():
                warn(
                    "[COLMAP] Source masks were detected, but they are not copied because the final "
                    "published dataset uses undistorted images."
                )
            elif dataset_masks_path.exists() and not _same_path(
                masks_source_path, dataset_masks_path
            ):
                shutil.rmtree(dataset_masks_path)

            _reset_directory(sparse_root_path)

            info(f"[COLMAP] Dataset root: {recon_root_path}")
            info(f"[COLMAP] Temporary images directory: {staged_images_path}")
            info(f"[COLMAP] Final images directory: {dataset_images_path}")
            info(f"[COLMAP] Prepared {staged_image_count} images for reconstruction")

            database_path = os.fspath(work_root_path / "database.db")

            # Clean up any corrupted database from previous failed runs, including WAL and SHM files
            for db_file in [
                database_path,
                f"{database_path}-wal",
                f"{database_path}-shm",
            ]:
                if os.path.exists(db_file):
                    try:
                        os.remove(db_file)
                        if db_file == database_path:
                            info(
                                f"[COLMAP] Removed existing database file: {database_path}"
                            )
                    except OSError as e:
                        warn(f"[COLMAP] Could not remove {db_file}: {e}")
                        if db_file == database_path:
                            raise RuntimeError(
                                f"Cannot remove old database file {database_path}: {e}. "
                                "The file may be in use or you may lack permissions."
                            )

            sparse_root = os.fspath(sparse_root_path)
            mapping_root = os.fspath(mapping_root_path)

            info(f"[COLMAP] Sparse reconstruction directory: {sparse_root_path}")

            # ================================================
            # FEATURE EXTRACTION
            # ================================================

            self._ensure_not_cancelled()
            self._update(ReconStage.FEATURE_EXTRACTION, 10.0, "Extracting features")

            info("[COLMAP] Extracting features")

            camera_mode = (
                CameraMode.SINGLE if self.params.single_camera else CameraMode.AUTO
            )

            # Build reader_options and extraction_options (newer API)
            reader_options = pycolmap.ImageReaderOptions()
            reader_options.camera_model = self.params.camera_model

            extraction_options = pycolmap.FeatureExtractionOptions()
            # Set SIFT type if available
            if hasattr(pycolmap, "FeatureExtractorType"):
                extraction_options.type = pycolmap.FeatureExtractorType.SIFT
            extraction_options.sift.max_num_features = sift_max_num_features

            # Try GPU first, fallback to CPU if needed
            _try_set_attr(extraction_options, "use_gpu", True)
            try:
                info("[COLMAP] Feature extraction using GPU")
                pycolmap.extract_features(
                    database_path=database_path,
                    image_path=os.fspath(staged_images_path),
                    camera_mode=camera_mode,
                    reader_options=reader_options,
                    extraction_options=extraction_options,
                )
                info("[COLMAP] Feature extraction finished using GPU")
            except Exception as exc:
                if "CUDA" in str(exc) or "gpu" in str(exc).lower():
                    warn(f"[COLMAP] GPU extraction failed ({exc}), retrying with CPU")
                    _try_set_attr(extraction_options, "use_gpu", False)
                    pycolmap.extract_features(
                        database_path=database_path,
                        image_path=os.fspath(staged_images_path),
                        camera_mode=camera_mode,
                        reader_options=reader_options,
                        extraction_options=extraction_options,
                    )
                    info("[COLMAP] Feature extraction finished using CPU")
                else:
                    raise
            _trim_process_memory()

            # ------------------------------------------------
            # MATCHING
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.MATCHING, 30.0, "Matching images")

            info("[COLMAP] Matching images")

            # Build matching options with GPU enabled
            matching_opts = pycolmap.FeatureMatchingOptions()
            _try_set_attr(matching_opts, "use_gpu", True)
            _try_set_attr(matching_opts, "max_num_matches", sift_max_num_matches)

            def _run_match(fn_name: str, match_kwargs: dict) -> None:
                """Run matcher with GPU fallback to CPU."""
                fn = getattr(pycolmap, fn_name, None)
                if fn is None:
                    raise RuntimeError(f"pycolmap.{fn_name} is unavailable")
                try:
                    fn(**match_kwargs)
                except Exception as exc:
                    if "CUDA" in str(exc) or "gpu" in str(exc).lower():
                        warn(f"[COLMAP] GPU matching failed ({exc}), retrying with CPU")
                        # Find and update the matching_options to disable GPU
                        if "matching_options" in match_kwargs:
                            opts = match_kwargs["matching_options"]
                            _try_set_attr(opts, "use_gpu", False)
                        fn(**match_kwargs)
                        info(f"[COLMAP] {fn_name} finished using CPU")
                    else:
                        raise

            if matcher_name == "sequential":
                pairing_opts = pycolmap.SequentialPairingOptions()
                _try_set_attr(pairing_opts, "loop_detection", False)
                match_kwargs = {
                    "database_path": database_path,
                    "matching_options": matching_opts,
                    "pairing_options": pairing_opts,
                }
                _run_match("match_sequential", match_kwargs)
            elif matcher_name == "exhaustive":
                pairing_opts = pycolmap.ExhaustivePairingOptions()
                _try_set_attr(pairing_opts, "block_size", exhaustive_block_size)
                match_kwargs = {
                    "database_path": database_path,
                    "matching_options": matching_opts,
                    "pairing_options": pairing_opts,
                }
                _run_match("match_exhaustive", match_kwargs)

            info("[COLMAP] Matching finished")
            del matching_opts
            _trim_process_memory()

            # ------------------------------------------------
            # MAPPING
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.MAPPING, 60.0, "Running SfM mapping")

            has_global_mapping = hasattr(pycolmap, "global_mapping")
            has_global_pipeline_options = hasattr(pycolmap, "GlobalPipelineOptions")
            has_global_mapper_options = hasattr(pycolmap, "GlobalMapperOptions")
            has_calibrate_view_graph = hasattr(pycolmap, "calibrate_view_graph")
            has_view_graph_calibration_options = hasattr(
                pycolmap, "ViewGraphCalibrationOptions"
            )
            info(f"[COLMAP] Running {reconstruction_mode} mapping")

            if (
                reconstruction_mode == "global"
                and self.params.use_view_graph_calibration
            ):
                if has_calibrate_view_graph:
                    info("[COLMAP] Running view graph calibration")

                    if has_view_graph_calibration_options:
                        vg_opts = pycolmap.ViewGraphCalibrationOptions()
                        _try_set_attr(vg_opts, "min_calibrated_pair_ratio", 0.1)
                    else:
                        vg_opts = None

                    if vg_opts is None:
                        calibrated = pycolmap.calibrate_view_graph(database_path)
                    else:
                        calibrated = pycolmap.calibrate_view_graph(
                            database_path,
                            options=vg_opts,
                        )
                    info(f"[COLMAP] View graph calibration result: {calibrated}")

            if reconstruction_mode == "global":
                if not has_global_mapping:
                    raise RuntimeError(
                        "This pycolmap build does not expose global_mapping(). "
                        "Choose Incremental mode or install a COLMAP 4.x build with GLOMAP support."
                    )
                if not has_global_pipeline_options:
                    raise RuntimeError(
                        "This pycolmap build does not expose GlobalPipelineOptions. "
                        "Choose Incremental mode or install a build with full GLOMAP support."
                    )
                global_opts = pycolmap.GlobalPipelineOptions()
                _try_set_attr(global_opts, "num_threads", num_threads)
                _try_set_attr(global_opts, "random_seed", 0)
                _try_set_attr(global_opts, "min_num_matches", 15)
                _try_set_attr(global_opts, "ignore_watermarks", False)
                _try_set_attr(global_opts, "decompose_relative_pose", True)
                if has_global_mapper_options:
                    mapper_opts = pycolmap.GlobalMapperOptions()
                    _try_set_attr(mapper_opts, "num_threads", num_threads)
                    _try_set_attr(
                        mapper_opts, "ba_num_iterations", ba_global_max_num_iterations
                    )
                    _try_set_attr(mapper_opts, "skip_bundle_adjustment", False)
                    # Enable GPU bundle adjustment for global mapping
                    _try_set_attr(mapper_opts, "ba_use_gpu", True)
                    _try_set_attr(global_opts, "mapper", mapper_opts)
                reconstructions = pycolmap.global_mapping(
                    database_path=database_path,
                    image_path=os.fspath(staged_images_path),
                    output_path=mapping_root,
                    options=global_opts,
                )
            else:
                pipeline_opts = pycolmap.IncrementalPipelineOptions()
                _try_set_attr(
                    pipeline_opts,
                    "ba_global_max_num_iterations",
                    ba_global_max_num_iterations,
                )
                if hasattr(pipeline_opts, "multiple_models"):
                    _try_set_attr(pipeline_opts, "multiple_models", False)
                if hasattr(pipeline_opts, "max_num_models"):
                    _try_set_attr(pipeline_opts, "max_num_models", 1)
                # Enable GPU bundle adjustment for incremental mapping
                _try_set_attr(pipeline_opts, "ba_use_gpu", True)
                _try_set_attr(pipeline_opts, "ba_gpu_index", self.params.gpu_index)

                reconstructions = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=os.fspath(staged_images_path),
                    output_path=mapping_root,
                    options=pipeline_opts,
                )

            if not reconstructions:
                raise RuntimeError("COLMAP produced no reconstruction")

            reconstruction = next(iter(reconstructions.values()))

            num_images = len(reconstruction.images)
            num_points = len(reconstruction.points3D)
            recon_metrics = _compute_recon_metrics(reconstruction)

            info(f"[COLMAP] Registered images: {num_images}")
            info(f"[COLMAP] Sparse points: {num_points}")
            info(
                "[COLMAP] Mean reprojection error: "
                f"{recon_metrics.mean_reprojection_error_px:.4f} px"
            )
            info(
                "[COLMAP] Median reprojection error: "
                f"{recon_metrics.median_reprojection_error_px:.4f} px"
            )
            info(
                "[COLMAP] 90th percentile error: "
                f"{recon_metrics.p90_reprojection_error_px:.4f} px"
            )
            if recon_metrics.good_ratio >= 0.0:
                info(
                    "[COLMAP] Points below 2.0 px: "
                    f"{recon_metrics.good_ratio * 100.0:.1f}%"
                )

            if num_images == 0:
                raise RuntimeError("COLMAP failed: 0 registered images")

            if hasattr(reconstruction, "write_text"):
                reconstruction.write_text(sparse_root)
                info(f"[COLMAP] Sparse model saved to {sparse_root} (text format)")
            else:
                reconstruction.write(sparse_root)
                info(f"[COLMAP] Sparse model saved to {sparse_root}")

            export_ply = getattr(reconstruction, "export_PLY", None)
            if callable(export_ply):
                points_ply_path = os.path.join(sparse_root, "points3D.ply")
                try:
                    export_ply(points_ply_path)
                    info(f"[COLMAP] Sparse point cloud exported to {points_ply_path}")
                except Exception as exc:
                    warn(f"[COLMAP] Could not export PLY: {exc}")

            self._ensure_not_cancelled()
            self._update(ReconStage.MAPPING, 90.0, "Publishing undistorted dataset")
            info(f"[COLMAP] Publishing undistorted dataset to {recon_root_path}")

            _undistort_dataset(
                sparse_model_path=sparse_root_path,
                source_images_path=staged_images_path,
                output_root_path=temp_publish_root_path,
            )

            if masks_source_path.is_dir():
                warn(
                    "[COLMAP] Masks were not published because they do not match the undistorted images."
                )

            _remove_path(dataset_images_path)
            shutil.move(
                os.fspath(temp_publish_root_path / "images"),
                os.fspath(dataset_images_path),
            )

            _remove_path(sparse_root_path)
            shutil.move(
                os.fspath(temp_publish_root_path / "sparse"),
                os.fspath(sparse_root_path),
            )

            final_dataset_path = recon_root_path
            final_images_path = dataset_images_path
            final_sparse_root = _resolve_sparse_model_path(sparse_root_path)
            info(f"[COLMAP] Final dataset root: {final_dataset_path}")
            info(f"[COLMAP] Final images directory: {final_images_path}")
            info(f"[COLMAP] Final sparse model: {final_sparse_root}")
            info(
                "[COLMAP] Prepared "
                f"{_count_files_in_directory(final_images_path, allowed_extensions=IMAGE_EXTENSIONS)} "
                "undistorted images for training"
            )

            # Drop the large in-memory reconstruction after saving to reduce peak RAM.
            del reconstruction
            del reconstructions
            _trim_process_memory()

            # ------------------------------------------------
            # DONE
            # ------------------------------------------------

            elapsed = time.time() - t0

            result = ReconResult(
                success=True,
                recon_dir=os.fspath(final_sparse_root),
                dataset_dir=os.fspath(final_dataset_path),
                elapsed_s=elapsed,
                metrics=recon_metrics,
            )

            with self._lock:
                self._result = result

            self._update(ReconStage.DONE, 100.0, "Finished")

            info(f"[COLMAP] Sparse reconstruction completed in {elapsed:.2f}s")

        except Exception as e:
            msg = str(e)
            if msg == "Cancelled" or self._check_cancelled():
                info("[COLMAP] Reconstruction cancelled")
                with self._lock:
                    self._result = ReconResult(success=False, error="Cancelled")
                self._update(ReconStage.CANCELLED, self.progress, "Cancelled")
                return

            error(f"[COLMAP] Error: {e}")

            self._update(ReconStage.ERROR, self.progress, msg)

            with self._lock:
                self._result = ReconResult(success=False, error=msg)
        finally:
            if work_root_path is not None:
                shutil.rmtree(work_root_path, ignore_errors=True)


class MainPanel(lf.ui.Panel):
    id = "colmap.main"
    label = "COLMAP Reconstruction"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 100
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 100

    def __init__(self):
        self._doc = None
        self._handle = None
        self._scrub_fields = ScrubFieldController(
            specs=SCRUB_FIELD_SPECS,
            get_value=self._get_scrub_field_value,
            set_value=self._set_scrub_field_value,
        )

        self.images_dir = ""
        self.params = ColmapParams()

        self._job: Optional[ColmapReconJob] = None
        self._last_result: Optional[ReconResult] = None

        self._preset_options = ["Low", "Normal", "Custom"]
        self._preset_name = "Normal"
        self._matchers = ["exhaustive", "sequential"]
        self._camera_models = ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "SIMPLE_PINHOLE"]
        self._apply_preset("Normal")

        self._last_running = False
        self._last_stage = ""
        self._last_status = ""
        self._last_progress = -1.0
        self._last_result_key = None
        self._last_loaded_result_key = None
        self._last_log_text = ""
        self._collapsed = {"instructions", "advanced"}

    def on_mount(self, doc):
        self._doc = doc
        self._scrub_fields.mount(doc)
        self._sync_section_states()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("colmap")
        if model is None:
            return

        model.bind("preset", lambda: self._preset_name, self._set_preset)
        model.bind("images_dir", lambda: self.images_dir, self._set_images_dir)
        model.bind(
            "camera_model", lambda: self.params.camera_model, self._set_camera_model
        )
        model.bind(
            "single_camera", lambda: self.params.single_camera, self._set_single_camera
        )
        model.bind("matcher", lambda: self.params.matcher, self._set_matcher)
        model.bind(
            "reconstruction_mode",
            lambda: self.params.reconstruction_mode,
            self._set_reconstruction_mode,
        )
        model.bind(
            "use_view_graph_calibration",
            lambda: self.params.use_view_graph_calibration,
            self._set_view_graph_calibration,
        )
        model.bind(
            "downsample_multiplier",
            lambda: self.params.downsample_multiplier,
            self._set_downsample_multiplier,
        )
        model.bind(
            "sift_max_num_features",
            lambda: self.params.sift_max_num_features,
            self._set_sift_max_num_features,
        )
        model.bind(
            "sift_max_num_matches",
            lambda: self.params.sift_max_num_matches,
            self._set_sift_max_num_matches,
        )
        model.bind(
            "exhaustive_block_size",
            lambda: self.params.exhaustive_block_size,
            self._set_exhaustive_block_size,
        )
        model.bind(
            "ba_global_max_num_iterations",
            lambda: self.params.ba_global_max_num_iterations,
            self._set_ba_global_max_num_iterations,
        )

        model.bind_func("has_images_dir", lambda: bool(self.images_dir.strip()))
        model.bind_func(
            "images_dir_text", lambda: self.images_dir or "No folder selected."
        )
        model.bind_func(
            "show_exhaustive_block_size", lambda: self.params.matcher == "exhaustive"
        )
        model.bind_func("preset_description", self._preset_description)
        model.bind_func("show_logs", self._show_logs)
        model.bind_func("live_log_text", self._live_log_text)
        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("stage_text", self._stage_text)
        model.bind_func("progress_value", self._progress_value)
        model.bind_func("progress_pct", self._progress_pct)
        model.bind_func("progress_status", self._progress_status)
        model.bind_func(
            "show_results",
            lambda: self._last_result is not None and self._last_result.success,
        )
        model.bind_func(
            "result_path",
            self._result_dataset_path,
        )
        model.bind_func(
            "result_time",
            lambda: (
                f"{self._last_result.elapsed_s:.1f}s"
                if self._last_result and self._last_result.success
                else ""
            ),
        )
        model.bind_func("result_sparse_points", self._result_sparse_points)
        model.bind_func("result_mean_error", self._result_mean_error)
        model.bind_func("result_median_error", self._result_median_error)
        model.bind_func("result_p90_error", self._result_p90_error)
        model.bind_func(
            "show_error",
            lambda: self._last_result is not None and not self._last_result.success,
        )
        model.bind_func(
            "error_text",
            lambda: (
                self._last_result.error or "Unknown error"
                if self._last_result and not self._last_result.success
                else ""
            ),
        )

        model.bind_event("browse_images", self._on_browse_images)
        model.bind_event("do_start", self._on_do_start)
        model.bind_event("do_cancel", self._on_do_cancel)
        model.bind_event("toggle_section", self._on_toggle_section)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        dirty = self._scrub_fields.sync_all()

        job_result = self._job.result if self._job else None
        job_result_key = self._result_key(job_result)
        if job_result_key is not None and job_result_key != self._last_result_key:
            self._last_result = job_result
            self._last_result_key = job_result_key
            if (
                job_result
                and job_result.success
                and job_result_key != self._last_loaded_result_key
            ):
                self._load_dataset_result(job_result)
                self._last_loaded_result_key = job_result_key
            self._dirty(
                "show_results",
                "result_path",
                "result_sparse_points",
                "result_time",
                "result_mean_error",
                "result_median_error",
                "result_p90_error",
                "show_error",
                "error_text",
            )
            dirty = True

        current_log_text = self._live_log_text()
        if current_log_text != self._last_log_text:
            self._last_log_text = current_log_text
            self._dirty("show_logs", "live_log_text")
            dirty = True

        running = self._is_running()
        if running != self._last_running:
            self._last_running = running
            self._dirty("show_idle", "show_running")
            dirty = True

        if self._job:
            stage = self._job.stage.value
            status = self._job.status
            progress = self._job.progress
            if (
                stage != self._last_stage
                or status != self._last_status
                or progress != self._last_progress
            ):
                self._last_stage = stage
                self._last_status = status
                self._last_progress = progress
                self._dirty(
                    "stage_text", "progress_value", "progress_pct", "progress_status"
                )
                dirty = True

        return dirty

    def on_unmount(self, doc):
        if self._job and self._job.is_running():
            self._job.cancel()
        doc.remove_data_model("colmap")
        self._scrub_fields.unmount()
        self._doc = None
        self._handle = None

    def _get_section_elements(self, name: str):
        if not self._doc:
            return None, None, None
        header = self._doc.get_element_by_id(f"hdr-{name}")
        arrow = self._doc.get_element_by_id(f"arrow-{name}")
        content = self._doc.get_element_by_id(f"sec-{name}")
        return header, arrow, content

    def _sync_section_states(self) -> None:
        for name in ("instructions", "setup", "advanced"):
            header, arrow, content = self._get_section_elements(name)
            expanded = name not in self._collapsed
            if content:
                content.set_class("collapsed", not expanded)
            if arrow:
                arrow.set_class("is-expanded", expanded)
            if header:
                header.set_class("is-expanded", expanded)

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        expanding = name in self._collapsed
        if expanding:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)

        header, arrow, content = self._get_section_elements(name)
        if content:
            content.set_class("collapsed", not expanding)
        if arrow:
            arrow.set_class("is-expanded", expanding)
        if header:
            header.set_class("is-expanded", expanding)

    def _dirty(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field_name in fields:
            self._handle.dirty(field_name)

    def _is_running(self) -> bool:
        return self._job is not None and self._job.is_running()

    @staticmethod
    def _result_key(result: Optional[ReconResult]):
        if result is None:
            return None
        return (
            result.success,
            result.recon_dir,
            result.dataset_dir,
            result.elapsed_s,
            result.error,
            None
            if result.metrics is None
            else (
                result.metrics.total_points,
                result.metrics.mean_reprojection_error_px,
                result.metrics.median_reprojection_error_px,
                result.metrics.p90_reprojection_error_px,
                result.metrics.good_ratio,
            ),
        )

    def _stage_text(self) -> str:
        if not self._job:
            return "Idle"
        return self._job.stage.value.replace("_", " ").title()

    def _progress_value(self) -> str:
        if not self._job:
            return "0"
        return f"{max(0.0, min(1.0, self._job.progress / 100.0)):.4f}"

    def _progress_pct(self) -> str:
        if not self._job:
            return "0%"
        return f"{int(self._job.progress)}%"

    def _progress_status(self) -> str:
        if not self._job:
            return ""
        return self._job.status or ""

    def _show_logs(self) -> bool:
        return bool(self._live_log_text())

    def _live_log_text(self) -> str:
        if not self._job:
            return ""
        return self._job.log_text

    def _result_metrics(self) -> Optional[ReconMetrics]:
        if not self._last_result or not self._last_result.success:
            return None
        return self._last_result.metrics

    def _result_dataset_path(self) -> str:
        if not self._last_result or not self._last_result.success:
            return ""
        return self._last_result.dataset_dir or self._last_result.recon_dir or ""

    def _load_dataset_result(self, result: ReconResult) -> None:
        dataset_dir = result.dataset_dir or ""
        if not dataset_dir:
            return
        try:
            lf.log.info(f"[COLMAP] Loading dataset: {dataset_dir}")
            lf.load_file(dataset_dir, is_dataset=True)
        except Exception as exc:
            lf.log.error(f"[COLMAP] Failed to load dataset {dataset_dir}: {exc}")

    def _result_sparse_points(self) -> str:
        metrics = self._result_metrics()
        return str(metrics.total_points) if metrics is not None else "0"

    def _result_mean_error(self) -> str:
        metrics = self._result_metrics()
        return (
            _format_metric_px(metrics.mean_reprojection_error_px)
            if metrics is not None
            else "N/A"
        )

    def _result_median_error(self) -> str:
        metrics = self._result_metrics()
        return (
            _format_metric_px(metrics.median_reprojection_error_px)
            if metrics is not None
            else "N/A"
        )

    def _result_p90_error(self) -> str:
        metrics = self._result_metrics()
        return (
            _format_metric_px(metrics.p90_reprojection_error_px)
            if metrics is not None
            else "N/A"
        )

    def _sync_choice_indices_from_params(self) -> None:
        self._matcher_idx = (
            self._matchers.index(self.params.matcher)
            if self.params.matcher in self._matchers
            else 0
        )
        self._camera_model_idx = (
            self._camera_models.index(self.params.camera_model)
            if self.params.camera_model in self._camera_models
            else 0
        )

    def _apply_preset(self, name: str) -> None:
        if name == "Low":
            self.params = ColmapParams(**PRESET_LOW_PARAMS)
            self._preset_name = "Low"
            self._sync_choice_indices_from_params()
            return
        if name == "Normal":
            self.params = ColmapParams(**PRESET_NORMAL_PARAMS)
            self._preset_name = "Normal"
            self._sync_choice_indices_from_params()

    def _set_custom_preset(self) -> None:
        if self._preset_name != "Custom":
            self._preset_name = "Custom"
            self._dirty("preset", "preset_description")

    def _dirty_all_params(self) -> None:
        self._dirty(
            "camera_model",
            "single_camera",
            "matcher",
            "reconstruction_mode",
            "use_view_graph_calibration",
            "downsample_multiplier",
            "sift_max_num_features",
            "sift_max_num_matches",
            "exhaustive_block_size",
            "ba_global_max_num_iterations",
            "show_exhaustive_block_size",
            "preset_description",
        )

    def _preset_description(self) -> str:
        if self._preset_name == "Low":
            return "Low uses the lightest settings and is the safest choice on constrained machines."
        if self._preset_name == "Normal":
            return "Normal balances quality and speed for typical reconstructions."
        return "Custom is active because one or more advanced settings differ from the presets."

    def _get_scrub_field_value(self, prop: str) -> float:
        if prop == "downsample_multiplier":
            return float(self.params.downsample_multiplier)
        if prop == "sift_max_num_features":
            return float(self.params.sift_max_num_features)
        if prop == "sift_max_num_matches":
            return float(self.params.sift_max_num_matches)
        if prop == "exhaustive_block_size":
            return float(self.params.exhaustive_block_size)
        if prop == "ba_global_max_num_iterations":
            return float(self.params.ba_global_max_num_iterations)
        raise KeyError(prop)

    def _set_scrub_field_value(self, prop: str, value: float) -> None:
        if prop == "downsample_multiplier":
            self._set_downsample_multiplier(value)
            return
        if prop == "sift_max_num_features":
            self._set_sift_max_num_features(value)
            return
        if prop == "sift_max_num_matches":
            self._set_sift_max_num_matches(value)
            return
        if prop == "exhaustive_block_size":
            self._set_exhaustive_block_size(value)
            return
        if prop == "ba_global_max_num_iterations":
            self._set_ba_global_max_num_iterations(value)
            return
        raise KeyError(prop)

    def _set_preset(self, value):
        value = str(value or "")
        if value == "Low":
            self._apply_preset("Low")
            self._dirty("preset", "preset_description")
            self._dirty_all_params()
            return
        if value == "Normal":
            self._apply_preset("Normal")
            self._dirty("preset", "preset_description")
            self._dirty_all_params()
            return
        if value == "Custom" and self._preset_name != "Custom":
            self._dirty("preset", "preset_description")

    def _set_images_dir(self, value):
        self.images_dir = str(value or "").strip()
        self._dirty("images_dir", "images_dir_text", "has_images_dir")

    def _set_camera_model(self, value):
        value = str(value or "")
        if value in self._camera_models and value != self.params.camera_model:
            self.params.camera_model = value
            self._camera_model_idx = self._camera_models.index(value)
            self._set_custom_preset()
            self._dirty("camera_model")

    def _set_single_camera(self, value):
        value = bool(value)
        if value != self.params.single_camera:
            self.params.single_camera = value
            self._set_custom_preset()
            self._dirty("single_camera")

    def _set_matcher(self, value):
        value = str(value or "")
        if value in self._matchers and value != self.params.matcher:
            self.params.matcher = value
            self._matcher_idx = self._matchers.index(value)
            self._set_custom_preset()
            self._dirty("matcher", "show_exhaustive_block_size")

    def _set_reconstruction_mode(self, value):
        value = str(value or "")
        if (
            value in ("incremental", "global")
            and value != self.params.reconstruction_mode
        ):
            self.params.reconstruction_mode = value
            self._set_custom_preset()
            self._dirty("reconstruction_mode")

    def _set_view_graph_calibration(self, value):
        enabled = bool(value)
        if enabled != self.params.use_view_graph_calibration:
            self.params.use_view_graph_calibration = enabled
            self._set_custom_preset()
            self._dirty("use_view_graph_calibration")

    def _set_downsample_multiplier(self, value):
        self._set_int_param("downsample_multiplier", value, 1, 8)

    def _set_sift_max_num_features(self, value):
        self._set_int_param("sift_max_num_features", value, 1024, 10000)

    def _set_sift_max_num_matches(self, value):
        self._set_int_param("sift_max_num_matches", value, 1024, 10000)

    def _set_exhaustive_block_size(self, value):
        self._set_int_param("exhaustive_block_size", value, 5, 50)

    def _set_ba_global_max_num_iterations(self, value):
        self._set_int_param("ba_global_max_num_iterations", value, 10, 100)

    def _set_int_param(self, name: str, value, min_value: int, max_value: int):
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return
        parsed = max(min_value, min(max_value, parsed))
        current = getattr(self.params, name)
        if parsed != current:
            setattr(self.params, name, parsed)
            self._set_custom_preset()
            self._dirty(name)

    def _on_browse_images(self, handle, event, args):
        del handle, event, args
        picked = lf.ui.open_folder_dialog(
            "Select image folder", self.images_dir or os.getcwd()
        )
        if picked:
            self.images_dir = picked
            self._dirty("images_dir", "images_dir_text", "has_images_dir")

    def _on_do_start(self, handle, event, args):
        del handle, event, args
        self._start_job()
        self._dirty(
            "show_idle",
            "show_running",
            "show_logs",
            "live_log_text",
            "show_results",
            "result_path",
            "result_sparse_points",
            "result_time",
            "result_mean_error",
            "result_median_error",
            "result_p90_error",
            "show_error",
            "error_text",
            "stage_text",
            "progress_value",
            "progress_pct",
            "progress_status",
        )

    def _on_do_cancel(self, handle, event, args):
        del handle, event, args
        if self._job and self._job.is_running():
            self._job.cancel()
            self._dirty("stage_text", "progress_status")

    def _start_job(self):
        images_dir = (self.images_dir or "").strip()
        if not images_dir:
            self._last_result = ReconResult(
                success=False, error="Please select an images folder"
            )
            self._last_result_key = self._result_key(self._last_result)
            return

        self._last_result = None
        self._last_result_key = None
        self._last_loaded_result_key = None
        self._last_log_text = ""

        # Snapshot params so UI edits during a run don't affect the active reconstruction.
        job_params = ColmapParams(**vars(self.params))
        self._job = ColmapReconJob(images_dir, job_params)
        self._job.start()
