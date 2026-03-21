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

- **Normal (Default)**: Incremental reconstruction, exhaustive matching, `Max Image Size=1600`, `Max Features=2048`, `Max Matches=2048`, `Exhaustive Block Size=15`, `BA Max Iterations=50`.
- **Low**: Incremental reconstruction, sequential matching, `Max Image Size=1200`, `Max Features=1536`, `Max Matches=1024`, `Exhaustive Block Size=10`, `BA Max Iterations=20`.

### Reconstruction Modes

- **Incremental**: Default and generally the safest option.
- **Global (GLOMAP)**: Available when supported by the installed `pycolmap` build.

### Reported Metrics

- **Mean Reprojection Error**
- **Median Reprojection Error**
- **90th Percentile Reprojection Error**

## Output

The plugin writes a sparse COLMAP workspace next to the selected image folder:

```text
<images_folder>/<folder_name>_reconstruction/
```

The final sparse model is saved here:

```text
<images_folder>/<folder_name>_reconstruction/sparse/0
```
