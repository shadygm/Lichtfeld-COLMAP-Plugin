"""COLMAP reconstruction + dataset import panel."""
from __future__ import annotations

import ctypes
import gc
import glob
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import onnxruntime

onnx_dir = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")

libs = glob.glob(os.path.join(onnx_dir, "libonnxruntime.so*"))
if not libs:
    raise RuntimeError(f"ONNXRuntime library not found in {onnx_dir}")

ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)

import lichtfeld as lf
import pycolmap
from pycolmap import CameraMode

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
    UNDISTORTING = "undistorting"
    IMPORTING = "importing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


SCRUB_FIELD_SPECS = {
    "sift_max_image_size": ScrubFieldSpec(
        min_value=800.0,
        max_value=2400.0,
        step=100.0,
        fmt="%d",
        data_type=int,
    ),
    "sift_max_num_features": ScrubFieldSpec(
        min_value=1024.0,
        max_value=4096.0,
        step=512.0,
        fmt="%d",
        data_type=int,
    ),
    "sift_max_num_matches": ScrubFieldSpec(
        min_value=1024.0,
        max_value=4096.0,
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
    sift_max_image_size=1200,
    sift_max_num_features=1024,
    matcher="sequential",
    sift_max_num_matches=1024,
    exhaustive_block_size=10,
    ba_global_max_num_iterations=20,
)

PRESET_NORMAL_PARAMS = dict(
    camera_model="OPENCV",
    single_camera=True,
    sift_max_image_size=1600,
    sift_max_num_features=2048,
    matcher="vocab_tree",
    sift_max_num_matches=2048,
    exhaustive_block_size=15,
    ba_global_max_num_iterations=50,
)


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


@dataclass
class ColmapParams:
    # Feature extraction
    camera_model: str = "OPENCV"
    single_camera: bool = True
    sift_max_image_size: int = 1600
    sift_max_num_features: int = 2048

    # Matching
    matcher: str = "vocab_tree"  # exhaustive | sequential | vocab_tree
    sift_max_num_matches: int = 2048
    exhaustive_block_size: int = 15

    # Mapping
    ba_global_max_num_iterations: int = 50

    # Undistortion / export
    undistort_output_type: str = "COLMAP"  # COLMAP (writes images/ + sparse/)


@dataclass
class ReconResult:
    success: bool
    recon_dir: Optional[str] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None


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

    def is_running(self) -> bool:
        return self.stage in (
            ReconStage.CHECKING,
            ReconStage.FEATURE_EXTRACTION,
            ReconStage.MATCHING,
            ReconStage.MAPPING,
            ReconStage.UNDISTORTING,
            ReconStage.IMPORTING,
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

    def _ensure_not_cancelled(self):
        if self._check_cancelled():
            raise RuntimeError("Cancelled")

    def _run(self):
        t0 = time.time()

        try:
            self._update(ReconStage.CHECKING, 2.0, "Checking inputs")

            images_dir = os.path.abspath(self.images_dir)

            if not os.path.isdir(images_dir):
                raise RuntimeError(f"Images directory does not exist: {images_dir}")

            # ------------------------------------------------
            # Count images
            # ------------------------------------------------

            image_files = [
                f for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG"))
            ]

            lf.log.info(f"[COLMAP] Found {len(image_files)} images")

            if len(image_files) < 2:
                raise RuntimeError("Need at least 2 images")

            sift_max_image_size = max(800, int(self.params.sift_max_image_size))
            sift_max_num_features = max(1024, int(self.params.sift_max_num_features))
            sift_max_num_matches = max(1024, int(self.params.sift_max_num_matches))
            exhaustive_block_size = max(5, int(self.params.exhaustive_block_size))
            ba_global_max_num_iterations = max(1, int(self.params.ba_global_max_num_iterations))
            matcher_name = self.params.matcher
            num_threads = min(8, os.cpu_count() or 4)

            lf.log.info(
                "[COLMAP] Reconstruction settings: "
                f"matcher={matcher_name}, "
                f"cpu_threads={num_threads}, "
                f"sift_max_image_size={sift_max_image_size}, "
                f"sift_max_num_features={sift_max_num_features}, "
                f"sift_max_num_matches={sift_max_num_matches}, "
                f"exhaustive_block_size={exhaustive_block_size}, "
                f"ba_global_max_num_iterations={ba_global_max_num_iterations}"
            )

            # ------------------------------------------------
            # Setup folders
            # ------------------------------------------------

            dataset_name = os.path.basename(os.path.normpath(images_dir))

            recon_root = os.path.join(
                images_dir,
                f"{dataset_name}_reconstruction"
            )
            os.makedirs(recon_root, exist_ok=True)

            # Validate recon_root is writable and disk has space
            if not os.path.isdir(recon_root):
                raise RuntimeError(f"Failed to create reconstruction directory: {recon_root}")

            if not os.access(recon_root, os.W_OK):
                raise RuntimeError(f"Reconstruction directory is not writable: {recon_root}")

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
                import shutil
                stat = shutil.disk_usage(recon_root)
                free_gb = stat.free / (1024 ** 3)
                if free_gb < 0.5:
                    lf.log.warn(f"[COLMAP] Low disk space: only {free_gb:.2f} GiB available")
            except Exception:
                pass  # Non-fatal if we can't check disk space

            database_path = os.path.join(recon_root, "database.db")

            # Clean up any corrupted database from previous failed runs, including WAL and SHM files
            for db_file in [database_path, f"{database_path}-wal", f"{database_path}-shm"]:
                if os.path.exists(db_file):
                    try:
                        os.remove(db_file)
                        if db_file == database_path:
                            lf.log.info(f"[COLMAP] Removed existing database file: {database_path}")
                    except OSError as e:
                        lf.log.warn(f"[COLMAP] Could not remove {db_file}: {e}")
                        if db_file == database_path:
                            raise RuntimeError(
                                f"Cannot remove old database file {database_path}: {e}. "
                                "The file may be in use or you may lack permissions."
                            )

            sparse_root = os.path.join(recon_root, "sparse")
            os.makedirs(sparse_root, exist_ok=True)

            undistorted_dir = os.path.join(recon_root, "dense")

            lf.log.info(f"[COLMAP] Reconstruction directory: {recon_root}")

            lf.log.info(f"[COLMAP] Using database path: {database_path}")

            # ================================================
            # FEATURE EXTRACTION
            # ================================================

            self._ensure_not_cancelled()
            self._update(ReconStage.FEATURE_EXTRACTION, 10.0, "Extracting features")

            lf.log.info("[COLMAP] Starting feature extraction")

            camera_mode = CameraMode.SINGLE if self.params.single_camera else CameraMode.AUTO

            # Build reader_options (for camera_model) and extraction_options (for SIFT settings)
            reader_opts = None
            extraction_opts = None
            extraction_gpu_requested = False
            extraction_gpu_used = False

            # Try ImageReaderOptions (newer API)
            if hasattr(pycolmap, "ImageReaderOptions"):
                reader_opts = pycolmap.ImageReaderOptions()
                _try_set_attr(reader_opts, "camera_model", self.params.camera_model)

            # Try FeatureExtractionOptions (newer API) or SiftExtractionOptions (older)
            if hasattr(pycolmap, "FeatureExtractionOptions"):
                extraction_opts = pycolmap.FeatureExtractionOptions()
                extraction_gpu_requested = _try_set_attr(extraction_opts, "use_gpu", True) or extraction_gpu_requested
                _try_set_attr(extraction_opts, "num_threads", num_threads)
                _try_set_attr(extraction_opts, "max_image_size", sift_max_image_size)
                _try_set_attr(extraction_opts, "max_num_features", sift_max_num_features)
            elif hasattr(pycolmap, "SiftExtractionOptions"):
                extraction_opts = pycolmap.SiftExtractionOptions()
                extraction_gpu_requested = _try_set_attr(extraction_opts, "use_gpu", True) or extraction_gpu_requested
                _try_set_attr(extraction_opts, "num_threads", num_threads)
                _try_set_attr(extraction_opts, "max_image_size", sift_max_image_size)
                _try_set_attr(extraction_opts, "max_num_features", sift_max_num_features)

            lf.log.info(
                "[COLMAP] Feature extraction compute mode: "
                + ("GPU requested" if extraction_gpu_requested else "CPU only")
            )

            # Build extract_kwargs based on available API
            extract_kwargs = dict(
                database_path=database_path,
                image_path=images_dir,
                camera_mode=camera_mode,
            )

            # Prefer new API: reader_options + extraction_options
            if reader_opts is not None:
                extract_kwargs["reader_options"] = reader_opts
            else:
                # Fallback: try old API with camera_model parameter
                extract_kwargs["camera_model"] = self.params.camera_model

            # Try newer extraction_options name first, fall back to sift_options
            if extraction_opts is not None:
                if hasattr(pycolmap, "FeatureExtractionOptions"):
                    extract_kwargs["extraction_options"] = extraction_opts
                else:
                    extract_kwargs["sift_options"] = extraction_opts

            def _call_extract() -> None:
                try:
                    pycolmap.extract_features(**extract_kwargs)
                except TypeError as e:
                    fallback_kwargs = dict(extract_kwargs)
                    for k in ("extraction_options", "sift_options", "reader_options", "camera_model"):
                        if k in fallback_kwargs:
                            fallback_kwargs.pop(k)
                            try:
                                pycolmap.extract_features(**fallback_kwargs)
                                return
                            except TypeError:
                                continue
                    lf.log.error(f"[COLMAP] extract_features() failed with error: {e}")
                    raise

            try:
                _call_extract()
                extraction_gpu_used = extraction_gpu_requested
            except Exception as exc:
                if extraction_gpu_requested and extraction_opts is not None and _try_set_attr(extraction_opts, "use_gpu", False):
                    lf.log.warn(
                        "[COLMAP] GPU feature extraction failed "
                        f"({exc}); retrying on CPU."
                    )
                    _call_extract()
                    extraction_gpu_used = False
                else:
                    raise

            lf.log.info(
                "[COLMAP] Feature extraction finished "
                f"using {'GPU' if extraction_gpu_used else 'CPU'}"
            )
            del reader_opts
            del extraction_opts
            del extract_kwargs
            _trim_process_memory()

            # ------------------------------------------------
            # MATCHING
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.MATCHING, 30.0, "Matching images")

            lf.log.info("[COLMAP] Starting matching")

            sift_matching_opts = None
            matching_gpu_requested = False
            matching_gpu_used = False
            if hasattr(pycolmap, "SiftMatchingOptions"):
                sift_matching_opts = pycolmap.SiftMatchingOptions()
                matching_gpu_requested = _try_set_attr(sift_matching_opts, "use_gpu", True) or matching_gpu_requested
                _try_set_attr(sift_matching_opts, "num_threads", num_threads)
                _try_set_attr(sift_matching_opts, "max_num_matches", sift_max_num_matches)

            lf.log.info(
                "[COLMAP] Matching compute mode: "
                + ("GPU requested" if matching_gpu_requested else "CPU only")
            )

            def _run_matcher(fn_name: str, match_kwargs: dict, drop_order: tuple[str, ...]) -> bool:
                fn = getattr(pycolmap, fn_name, None)
                if fn is None:
                    return False
                try:
                    fn(**match_kwargs)
                    return True
                except TypeError:
                    fallback_kwargs = dict(match_kwargs)
                    for key in drop_order:
                        if key in fallback_kwargs:
                            fallback_kwargs.pop(key, None)
                            try:
                                fn(**fallback_kwargs)
                                return True
                            except TypeError:
                                continue
                    raise

            def _run_matcher_with_gpu_fallback(fn_name: str, match_kwargs: dict, drop_order: tuple[str, ...]) -> bool:
                nonlocal matching_gpu_used
                try:
                    result = _run_matcher(fn_name, match_kwargs, drop_order)
                    matching_gpu_used = matching_gpu_requested
                    return result
                except Exception as exc:
                    if (
                        matching_gpu_requested
                        and sift_matching_opts is not None
                        and _try_set_attr(sift_matching_opts, "use_gpu", False)
                    ):
                        lf.log.warn(
                            f"[COLMAP] GPU matching failed in {fn_name} "
                            f"({exc}); retrying on CPU."
                        )
                        result = _run_matcher(fn_name, match_kwargs, drop_order)
                        matching_gpu_used = False
                        return result
                    raise

            if matcher_name == "vocab_tree":
                match_kwargs = {"database_path": database_path}
                if sift_matching_opts is not None:
                    match_kwargs["sift_options"] = sift_matching_opts
                vocab_opts = None
                if hasattr(pycolmap, "VocabTreeMatchingOptions"):
                    vocab_opts = pycolmap.VocabTreeMatchingOptions()
                    _try_set_attr(vocab_opts, "num_images", 100)
                    _try_set_attr(vocab_opts, "num_nearest_neighbors", 5)
                    match_kwargs["matching_options"] = vocab_opts
                if hasattr(pycolmap, "match_vocab_tree"):
                    try:
                        if not _run_matcher_with_gpu_fallback(
                            "match_vocab_tree",
                            match_kwargs,
                            ("matching_options", "sift_options"),
                        ):
                            raise RuntimeError("pycolmap.match_vocab_tree is unavailable")
                    except Exception as exc:
                        lf.log.warn(
                            "[COLMAP] Vocab tree matching unavailable or failed "
                            f"({exc}); falling back to sequential matching."
                        )
                        matcher_name = "sequential"
                else:
                    lf.log.warn(
                        "[COLMAP] pycolmap.match_vocab_tree is unavailable; "
                        "falling back to sequential matching."
                    )
                    matcher_name = "sequential"

            if matcher_name == "sequential":
                match_kwargs = {"database_path": database_path}
                if sift_matching_opts is not None:
                    match_kwargs["sift_options"] = sift_matching_opts
                if hasattr(pycolmap, "SequentialMatchingOptions"):
                    match_kwargs["matching_options"] = pycolmap.SequentialMatchingOptions()
                if not _run_matcher_with_gpu_fallback(
                    "match_sequential",
                    match_kwargs,
                    ("matching_options", "sift_options"),
                ):
                    raise RuntimeError("pycolmap.match_sequential is unavailable")
            elif matcher_name == "exhaustive":
                match_kwargs = {"database_path": database_path}
                if sift_matching_opts is not None:
                    match_kwargs["sift_options"] = sift_matching_opts
                if hasattr(pycolmap, "ExhaustiveMatchingOptions"):
                    exhaustive_opts = pycolmap.ExhaustiveMatchingOptions()
                    _try_set_attr(exhaustive_opts, "block_size", exhaustive_block_size)
                    match_kwargs["matching_options"] = exhaustive_opts
                if not _run_matcher_with_gpu_fallback(
                    "match_exhaustive",
                    match_kwargs,
                    ("matching_options", "sift_options"),
                ):
                    raise RuntimeError("pycolmap.match_exhaustive is unavailable")

            lf.log.info(
                "[COLMAP] Matching finished "
                f"using {'GPU' if matching_gpu_used else 'CPU'}"
            )
            del sift_matching_opts
            _trim_process_memory()

            # ------------------------------------------------
            # MAPPING
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.MAPPING, 60.0, "Running SfM mapping")

            lf.log.info("[COLMAP] Starting incremental mapping")

            pipeline_opts = pycolmap.IncrementalPipelineOptions()
            _try_set_attr(pipeline_opts, "ba_global_max_num_iterations", ba_global_max_num_iterations)
            if hasattr(pipeline_opts, "multiple_models"):
                _try_set_attr(pipeline_opts, "multiple_models", False)
            if hasattr(pipeline_opts, "max_num_models"):
                _try_set_attr(pipeline_opts, "max_num_models", 1)

            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=sparse_root,
                options=pipeline_opts,
            )

            if not reconstructions:
                raise RuntimeError("COLMAP produced no reconstruction")

            reconstruction = next(iter(reconstructions.values()))

            num_images = len(reconstruction.images)
            num_points = len(reconstruction.points3D)

            lf.log.info(f"[COLMAP] Registered images: {num_images}")
            lf.log.info(f"[COLMAP] Sparse points: {num_points}")

            if num_images == 0:
                raise RuntimeError("COLMAP failed: 0 registered images")

            sparse_model_dir = os.path.join(sparse_root, "0")
            os.makedirs(sparse_model_dir, exist_ok=True)

            reconstruction.write(sparse_model_dir)

            lf.log.info(f"[COLMAP] Sparse model saved to {sparse_model_dir}")

            # Drop the large in-memory reconstruction before undistortion to reduce peak RAM.
            del reconstruction
            del reconstructions
            _trim_process_memory()

            # ------------------------------------------------
            # UNDISTORTION
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.UNDISTORTING, 85.0, "Exporting dataset")

            sparse_model_dir = os.path.join(sparse_root, "0")

            pycolmap.undistort_images(
                output_path=undistorted_dir,
                input_path=sparse_model_dir,
                image_path=images_dir,
                output_type=self.params.undistort_output_type,
            )

            lf.log.info("[COLMAP] Undistortion finished")

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            images_dir_check = os.path.join(undistorted_dir, "images")
            sparse_dir_check = os.path.join(undistorted_dir, "sparse")

            if not os.path.isdir(images_dir_check) or not os.path.isdir(sparse_dir_check):
                raise RuntimeError("COLMAP export invalid: missing images/ or sparse/")

            lf.log.info("[COLMAP] Dataset export valid")

            # ------------------------------------------------
            # DONE
            # ------------------------------------------------

            elapsed = time.time() - t0

            result = ReconResult(
                success=True,
                recon_dir=undistorted_dir,
                elapsed_s=elapsed,
            )

            with self._lock:
                self._result = result

            self._update(ReconStage.DONE, 100.0, "Finished")

            lf.log.info(f"[COLMAP] Reconstruction completed in {elapsed:.2f}s")

        except Exception as e:
            msg = str(e)
            if msg == "Cancelled" or self._check_cancelled():
                lf.log.info("[COLMAP] Reconstruction cancelled")
                with self._lock:
                    self._result = ReconResult(success=False, error="Cancelled")
                self._update(ReconStage.CANCELLED, self.progress, "Cancelled")
                return

            lf.log.error(f"[COLMAP] Error: {e}")

            self._update(ReconStage.ERROR, self.progress, msg)

            with self._lock:
                self._result = ReconResult(success=False, error=msg)


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
        self._pending_import_path: Optional[str] = None

        self._preset_options = ["Low", "Normal", "Custom"]
        self._preset_name = "Low"
        self._matchers = ["exhaustive", "sequential", "vocab_tree"]
        self._camera_models = ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "SIMPLE_PINHOLE"]
        self._undistort_type_idx = 0
        self._undistort_types = ["COLMAP"]
        self._apply_preset("Low")

        self._auto_import = True

        self._last_running = False
        self._last_stage = ""
        self._last_status = ""
        self._last_progress = -1.0
        self._last_result_key = None
        self._cached_image_count_dir = None
        self._cached_image_count = 0
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
        model.bind("camera_model", lambda: self.params.camera_model, self._set_camera_model)
        model.bind("single_camera", lambda: self.params.single_camera, self._set_single_camera)
        model.bind("matcher", lambda: self.params.matcher, self._set_matcher)
        model.bind(
            "sift_max_image_size",
            lambda: str(self.params.sift_max_image_size),
            self._set_sift_max_image_size,
        )
        model.bind(
            "sift_max_num_features",
            lambda: str(self.params.sift_max_num_features),
            self._set_sift_max_num_features,
        )
        model.bind(
            "sift_max_num_matches",
            lambda: str(self.params.sift_max_num_matches),
            self._set_sift_max_num_matches,
        )
        model.bind(
            "exhaustive_block_size",
            lambda: str(self.params.exhaustive_block_size),
            self._set_exhaustive_block_size,
        )
        model.bind(
            "ba_global_max_num_iterations",
            lambda: str(self.params.ba_global_max_num_iterations),
            self._set_ba_global_max_num_iterations,
        )
        model.bind(
            "undistort_output_type",
            lambda: self.params.undistort_output_type,
            self._set_undistort_output_type,
        )
        model.bind("auto_import", lambda: self._auto_import, self._set_auto_import)

        model.bind_func("has_images_dir", lambda: bool(self.images_dir.strip()))
        model.bind_func("images_dir_text", lambda: self.images_dir or "No folder selected.")
        model.bind_func("show_exhaustive_block_size", lambda: self.params.matcher == "exhaustive")
        model.bind_func("preset_description", self._preset_description)
        model.bind_func("show_config_warning", self._show_config_warning)
        model.bind_func("warning_text", self._warning_text)
        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("stage_text", self._stage_text)
        model.bind_func("progress_value", self._progress_value)
        model.bind_func("progress_pct", self._progress_pct)
        model.bind_func("progress_status", self._progress_status)
        model.bind_func(
            "can_manual_import",
            lambda: (
                self._last_result is not None
                and self._last_result.success
                and not self._auto_import
                and bool(self._last_result.recon_dir)
            ),
        )
        model.bind_func(
            "show_results",
            lambda: self._last_result is not None and self._last_result.success,
        )
        model.bind_func(
            "result_path",
            lambda: self._last_result.recon_dir or ""
            if self._last_result and self._last_result.success
            else "",
        )
        model.bind_func(
            "result_time",
            lambda: f"{self._last_result.elapsed_s:.1f}s"
            if self._last_result and self._last_result.success
            else "",
        )
        model.bind_func(
            "show_error",
            lambda: self._last_result is not None and not self._last_result.success,
        )
        model.bind_func(
            "error_text",
            lambda: self._last_result.error or "Unknown error"
            if self._last_result and not self._last_result.success
            else "",
        )

        model.bind_event("browse_images", self._on_browse_images)
        model.bind_event("do_start", self._on_do_start)
        model.bind_event("do_cancel", self._on_do_cancel)
        model.bind_event("do_import", self._on_do_import)
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
                job_result.success
                and self._auto_import
                and job_result.recon_dir
                and self._pending_import_path is None
            ):
                self._pending_import_path = job_result.recon_dir
            self._dirty(
                "show_results",
                "result_path",
                "result_time",
                "show_error",
                "error_text",
                "can_manual_import",
            )
            dirty = True

        if self._pending_import_path:
            path = self._pending_import_path
            self._pending_import_path = None
            if self._job:
                with self._job._lock:
                    self._job._stage = ReconStage.IMPORTING
                    self._job._progress = 98.0
                    self._job._status = "Importing dataset into LichtFeld..."
            try:
                lf.load_file(path, is_dataset=True)
                lf.log.info(f"Imported dataset: {path}")
                if self._job:
                    with self._job._lock:
                        self._job._stage = ReconStage.DONE
                        self._job._progress = 100.0
                        self._job._status = "Imported"
            except Exception as exc:
                lf.log.error(f"Failed to import dataset: {exc}")
                if self._job:
                    with self._job._lock:
                        self._job._stage = ReconStage.ERROR
                        self._job._status = f"Import failed: {exc}"
            self._dirty(
                "show_idle",
                "show_running",
                "stage_text",
                "progress_value",
                "progress_pct",
                "progress_status",
                "show_results",
                "show_error",
                "error_text",
                "can_manual_import",
            )
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
                self._dirty("stage_text", "progress_value", "progress_pct", "progress_status")
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
            result.elapsed_s,
            result.error,
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

    def _show_config_warning(self) -> bool:
        return (
            self.params.sift_max_num_features > 4096
            or self.params.sift_max_num_matches > 4096
            or self.params.matcher == "exhaustive"
        )

    def _warning_text(self) -> str:
        reasons = []
        if self.params.sift_max_num_features > 4096:
            reasons.append("feature count is above 4096")
        if self.params.sift_max_num_matches > 4096:
            reasons.append("match count is above 4096")
        if self.params.matcher == "exhaustive":
            reasons.append("exhaustive matching scales poorly on large image sets")
            image_count = self._selected_image_count()
            if image_count > 100:
                reasons.append(f"the selected folder has {image_count} images")
        if not reasons:
            return ""
        return "Warning: higher-memory configuration enabled because " + "; ".join(reasons) + "."

    def _selected_image_count(self) -> int:
        images_dir = (self.images_dir or "").strip()
        if images_dir == self._cached_image_count_dir:
            return self._cached_image_count
        if not images_dir or not os.path.isdir(images_dir):
            self._cached_image_count_dir = images_dir
            self._cached_image_count = 0
            return 0
        try:
            count = sum(
                1
                for name in os.listdir(images_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        except Exception:
            count = 0
        self._cached_image_count_dir = images_dir
        self._cached_image_count = count
        return count

    def _sync_choice_indices_from_params(self) -> None:
        self._matcher_idx = self._matchers.index(self.params.matcher) if self.params.matcher in self._matchers else 0
        self._camera_model_idx = (
            self._camera_models.index(self.params.camera_model)
            if self.params.camera_model in self._camera_models
            else 0
        )
        self._undistort_type_idx = (
            self._undistort_types.index(self.params.undistort_output_type)
            if self.params.undistort_output_type in self._undistort_types
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
            "sift_max_image_size",
            "sift_max_num_features",
            "sift_max_num_matches",
            "exhaustive_block_size",
            "ba_global_max_num_iterations",
            "show_exhaustive_block_size",
            "preset_description",
            "show_config_warning",
            "warning_text",
        )

    def _preset_description(self) -> str:
        if self._preset_name == "Low":
            return "Low uses the lightest settings and is the safest choice on constrained machines."
        if self._preset_name == "Normal":
            return "Normal balances quality and speed for typical reconstructions."
        return "Custom is active because one or more advanced settings differ from the presets."

    def _get_scrub_field_value(self, prop: str) -> float:
        if prop == "sift_max_image_size":
            return float(self.params.sift_max_image_size)
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
        if prop == "sift_max_image_size":
            self._set_sift_max_image_size(value)
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
        self._cached_image_count_dir = None
        self._cached_image_count = 0
        self._dirty("images_dir", "images_dir_text", "has_images_dir", "show_config_warning", "warning_text")

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
            self._dirty("matcher", "show_exhaustive_block_size", "show_config_warning", "warning_text")

    def _set_sift_max_image_size(self, value):
        self._set_int_param("sift_max_image_size", value, 800, 2400)

    def _set_sift_max_num_features(self, value):
        self._set_int_param("sift_max_num_features", value, 1024, 4096)

    def _set_sift_max_num_matches(self, value):
        self._set_int_param("sift_max_num_matches", value, 1024, 4096)

    def _set_exhaustive_block_size(self, value):
        self._set_int_param("exhaustive_block_size", value, 5, 50)

    def _set_ba_global_max_num_iterations(self, value):
        self._set_int_param("ba_global_max_num_iterations", value, 10, 100)

    def _set_undistort_output_type(self, value):
        value = str(value or "")
        if value in self._undistort_types:
            self.params.undistort_output_type = value
            self._undistort_type_idx = self._undistort_types.index(value)
            self._dirty("undistort_output_type")

    def _set_auto_import(self, value):
        self._auto_import = bool(value)
        self._dirty("auto_import", "can_manual_import")

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
            self._dirty(name, "show_config_warning", "warning_text")

    def _on_browse_images(self, handle, event, args):
        del handle, event, args
        picked = lf.ui.open_folder_dialog("Select image folder", self.images_dir or os.getcwd())
        if picked:
            self.images_dir = picked
            self._cached_image_count_dir = None
            self._cached_image_count = 0
            self._dirty("images_dir", "images_dir_text", "has_images_dir", "show_config_warning", "warning_text")

    def _on_do_start(self, handle, event, args):
        del handle, event, args
        self._start_job()
        self._dirty(
            "show_idle",
            "show_running",
            "show_results",
            "show_error",
            "error_text",
            "stage_text",
            "progress_value",
            "progress_pct",
            "progress_status",
            "can_manual_import",
        )

    def _on_do_cancel(self, handle, event, args):
        del handle, event, args
        if self._job and self._job.is_running():
            self._job.cancel()
            self._dirty("stage_text", "progress_status")

    def _on_do_import(self, handle, event, args):
        del handle, event, args
        if self._last_result and self._last_result.success and self._last_result.recon_dir:
            self._pending_import_path = self._last_result.recon_dir
            self._dirty("can_manual_import")

    def _start_job(self):
        images_dir = (self.images_dir or "").strip()
        if not images_dir:
            self._last_result = ReconResult(success=False, error="Please select an images folder")
            self._last_result_key = self._result_key(self._last_result)
            return

        self._last_result = None
        self._last_result_key = None
        self._pending_import_path = None

        # Snapshot params so UI edits during a run don't affect the active reconstruction.
        job_params = ColmapParams(**vars(self.params))
        self._job = ColmapReconJob(images_dir, job_params)
        self._job.start()
