"""COLMAP reconstruction + dataset import panel."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import lichtfeld as lf
import pycolmap
from pycolmap import CameraMode
from lfs_plugins.types import Panel


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


@dataclass
class ColmapParams:
    # Feature extraction
    camera_model: str = "OPENCV"
    single_camera: bool = True

    # Matching
    matcher: str = "exhaustive"  # exhaustive | sequential

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
        on_progress: Optional[Callable[[ReconStage, float, str], None]] = None,
        on_complete: Optional[Callable[[ReconResult], None]] = None,
    ):
        self.images_dir = images_dir
        self.params = params
        self.on_progress = on_progress
        self.on_complete = on_complete

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

        if self.on_progress:
            self.on_progress(stage, progress, status)

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

            # ------------------------------------------------
            # Setup folders
            # ------------------------------------------------

            dataset_name = os.path.basename(os.path.normpath(images_dir))

            recon_root = os.path.join(
                images_dir,
                f"{dataset_name}_reconstruction"
            )
            os.makedirs(recon_root, exist_ok=True)

            database_path = os.path.join(recon_root, "database.db")

            sparse_root = os.path.join(recon_root, "sparse")
            os.makedirs(sparse_root, exist_ok=True)

            undistorted_dir = os.path.join(recon_root, "dense")

            lf.log.info(f"[COLMAP] Reconstruction directory: {recon_root}")

            # ------------------------------------------------
            # FEATURE EXTRACTION
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.FEATURE_EXTRACTION, 10.0, "Extracting features")

            lf.log.info("[COLMAP] Starting feature extraction")

            camera_mode = CameraMode.SINGLE if self.params.single_camera else CameraMode.AUTO

            pycolmap.extract_features(
                database_path=database_path,
                image_path=images_dir,
                camera_model=self.params.camera_model,
                camera_mode=camera_mode,
            )

            lf.log.info("[COLMAP] Feature extraction finished")

            # ------------------------------------------------
            # MATCHING
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.MATCHING, 30.0, "Matching images")

            lf.log.info("[COLMAP] Starting matching")

            if self.params.matcher == "sequential":
                pycolmap.match_sequential(database_path=database_path)
            else:
                pycolmap.match_exhaustive(database_path=database_path)

            lf.log.info("[COLMAP] Matching finished")

            # ------------------------------------------------
            # MAPPING
            # ------------------------------------------------

            self._ensure_not_cancelled()
            self._update(ReconStage.MAPPING, 60.0, "Running SfM mapping")

            lf.log.info("[COLMAP] Starting incremental mapping")

            pipeline_opts = pycolmap.IncrementalPipelineOptions()

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
                output_type="COLMAP",
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

            if self.on_complete:
                self.on_complete(result)

        except Exception as e:

            lf.log.error(f"[COLMAP] Error: {e}")

            msg = str(e)

            self._update(ReconStage.ERROR, self.progress, msg)

            with self._lock:
                self._result = ReconResult(success=False, error=msg)


class MainPanel(Panel):
    label = "COLMAP Reconstruction"
    space = "MAIN_PANEL_TAB"
    order = 100

    def __init__(self):
        self.images_dir = ""
        self.params = ColmapParams()

        self._job: Optional[ColmapReconJob] = None
        self._last_result: Optional[ReconResult] = None
        self._pending_import_path: Optional[str] = None

        self._matcher_idx = 0
        self._matchers = ["exhaustive", "sequential"]

        self._camera_models = ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "SIMPLE_PINHOLE"]
        self._camera_model_idx = 0
        self.params.camera_model = self._camera_models[self._camera_model_idx]

        self._undistort_type_idx = 0
        self._undistort_types = ["COLMAP"]

        self._auto_import = True

    def draw(self, ui):
        # Import must happen on main thread.
        if self._pending_import_path:
            path = self._pending_import_path
            self._pending_import_path = None
            # Update UI progress state without triggering recursion.
            if self._job:
                with self._job._lock:
                    self._job._stage = ReconStage.IMPORTING
                    self._job._progress = 98.0
                    self._job._status = "Importing dataset into LFS..."
            try:
                lf.load_file(path, is_dataset=True)
                lf.log.info(f"Imported dataset: {path}")
                if self._job:
                    with self._job._lock:
                        self._job._stage = ReconStage.DONE
                        self._job._progress = 100.0
                        self._job._status = "Imported"
            except Exception as e:
                lf.log.error(f"Failed to import dataset: {e}")
                if self._job:
                    with self._job._lock:
                        self._job._stage = ReconStage.ERROR
                        self._job._status = f"Import failed: {e}"

        ui.heading("COLMAP Reconstruction")
        ui.label("Run COLMAP on an image folder, then load result as a dataset")
        ui.separator()

        _, self.images_dir = ui.path_input("Images Folder", self.images_dir, folder_mode=True)
        ui.same_line()
        if ui.small_button("Browse"):
            picked = lf.ui.open_folder_dialog("Select image folder", self.images_dir or os.getcwd())
            if picked:
                self.images_dir = picked

        ui.separator()

        if ui.collapsing_header("Basic Parameters", default_open=True):
            _, self._camera_model_idx = ui.combo("Camera Model", self._camera_model_idx, self._camera_models)
            self.params.camera_model = self._camera_models[self._camera_model_idx]

            _, self.params.single_camera = ui.checkbox("Single Camera", self.params.single_camera)

            _, self._matcher_idx = ui.combo("Matcher", self._matcher_idx, self._matchers)
            self.params.matcher = self._matchers[self._matcher_idx]

            _, self.params.ba_global_max_num_iterations = ui.drag_int(
                "BA Global Max Iters",
                self.params.ba_global_max_num_iterations,
                1,
                10,
                500,
            )

            _, self._undistort_type_idx = ui.combo("Export Type", self._undistort_type_idx, self._undistort_types)
            self.params.undistort_output_type = self._undistort_types[self._undistort_type_idx]

        ui.separator()
        _, self._auto_import = ui.checkbox("Auto-import into scene", self._auto_import)
        ui.separator()

        if self._job and self._job.is_running():
            stage = self._job.stage.value
            progress = self._job.progress
            ui.label(f"Stage: {stage}")
            ui.progress_bar(progress / 100.0, self._job.status)
            if ui.button("Cancel"):
                self._job.cancel()
            return

        start_disabled = bool(self._job and self._job.is_running())
        if start_disabled:
            ui.begin_disabled()
        if ui.button_styled("Run Reconstruction", "primary", (0, 36)):
            self._start_job()
        if start_disabled:
            ui.end_disabled()

        if self._last_result:
            ui.separator()
            if self._last_result.success:
                ui.heading("Result")
                ui.label(f"Output: {self._last_result.recon_dir}")
                ui.label(f"Time: {self._last_result.elapsed_s:.1f}s")
                if not self._auto_import and self._last_result.recon_dir:
                    if ui.button("Import Dataset", (0, 36)):
                        self._pending_import_path = self._last_result.recon_dir
            else:
                ui.text_colored("Error:", (1.0, 0.3, 0.3, 1.0))
                ui.text_selectable(self._last_result.error or "Unknown error", 80)

    def _start_job(self):
        images_dir = (self.images_dir or "").strip()
        if not images_dir:
            self._last_result = ReconResult(success=False, error="Please select an images folder")
            return

        self._last_result = None

        def on_progress(stage: ReconStage, pct: float, msg: str):
            # Job already updates its own state; this hook is only for optional side effects.
            # (Do NOT call job._update() here, it would recurse.)
            pass

        def on_complete(res: ReconResult):
            self._last_result = res
            if res.success and self._auto_import and res.recon_dir:
                self._pending_import_path = res.recon_dir

        self._job = ColmapReconJob(images_dir, self.params, on_progress=on_progress, on_complete=on_complete)
        self._job.start()
