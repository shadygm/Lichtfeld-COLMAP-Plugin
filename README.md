# COLMAP Reconstruction Plugin for LichtFeld Studio

A sparse reconstruction plugin for LichtFeld Studio. It runs COLMAP via `pycolmap` on an image folder, builds a sparse SfM model, and reports reconstruction metrics directly in the panel.

## Features

- **Sparse COLMAP Reconstruction**
- **Incremental and Global Mapping Modes**
- **Reprojection Error Metrics in UI**

## Installation

### Manual Installation

```bash
git clone https://github.com/shadygm/Lichtfeld-COLMAP-Plugin.git ~/.lichtfeld/plugins/Lichtfeld-COLMAP-Plugin
```

## Usage

### GUI

1. Open the **COLMAP Reconstruction** panel in LichtFeld Studio.
2. Select a folder containing overlapping `JPG` or `PNG` images.
3. Leave **Normal** selected for the default workflow, or switch to **Low** for a lighter setup.
4. Optional: Open **Advanced** to adjust matcher, reconstruction mode, camera model, and optimization parameters.
5. Click **Run Reconstruction**.
6. Monitor the stage, progress bar, and live logs while COLMAP runs.
7. Review the output path and reprojection error statistics when reconstruction completes.

## Configuration

### Presets

- **Normal (Default)**: Incremental reconstruction, exhaustive matching, `Downsample=1x (Full Resolution)`, `Max Features=2048`, `Max Matches=2048`, `Exhaustive Block Size=15`, `BA Max Iterations=50`.
- **Low**: Incremental reconstruction, sequential matching, `Downsample=2x (Half Resolution)`, `Max Features=1536`, `Max Matches=1024`, `Exhaustive Block Size=10`, `BA Max Iterations=20`.

### Downsample Multiplier

The plugin applies a downsample multiplier to input images before reconstruction:

| Multiplier | Resolution | Use Case |
|------------|------------|----------|
| **1x** | Full resolution | Best quality, slower processing |
| **2x** | Half resolution (50%) | Balanced quality and speed |
| **4x** | Quarter resolution (25%) | Faster processing, lower quality |
| **8x** | Eighth resolution (12.5%) | Fastest processing, draft quality |

Images are pre-resized using high-quality Lanczos resampling before feature extraction. The downsampled images are saved as JPEG (quality 95) in the working directory.

### Reconstruction Modes

- **Incremental**: Default and generally the safest option.
- **Global (GLOMAP)**: Available when supported by the installed `pycolmap` build.

### Matching Modes

- **Exhaustive**: Matches all image pairs. This is the default and is usually best for small to medium image folders.
- **Sequential**: Matches neighboring filenames for video-like image sequences. Use sequentially ordered names such as `image0001.jpg`, `image0002.jpg`, etc.

### Reported Metrics

- **Mean Reprojection Error**
- **Median Reprojection Error**
- **90th Percentile Reprojection Error**

## Output

The plugin writes a standard COLMAP-style dataset layout. If you select an existing `images/` folder, the dataset root is its parent directory. Otherwise, the plugin creates a sibling dataset folder named after the selected image folder:

```text
<dataset_root>/
├── images/
└── sparse/
```

After a successful reconstruction, the plugin loads `<dataset_root>/` into LichtFeld as the dataset for training.
