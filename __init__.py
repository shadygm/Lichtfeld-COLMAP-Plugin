"""
Lichtfeld-COLMAP-Plugin - A LichtFeld Studio plugin.
"""

import os
import sys
from pathlib import Path

# Ensure plugin venv site-packages is first in path
_venv_site_packages = (
    Path(__file__).parent / ".venv" / "lib" / "python3.12" / "site-packages"
)

# Remove other plugin venv paths to avoid conflicts
for p in [p for p in sys.path if ".venv" in p and str(_venv_site_packages) not in p]:
    sys.path.remove(p)

# Insert our path at the beginning
if str(_venv_site_packages) in sys.path:
    sys.path.remove(str(_venv_site_packages))
sys.path.insert(0, str(_venv_site_packages))

# Clear any cached pycolmap imports
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("pycolmap"):
        del sys.modules[mod_name]

# Preload CUDA libraries so pycolmap can find them
_venv_libs = _venv_site_packages / "pycolmap.libs"
if _venv_libs.exists():
    import ctypes

    for lib in [
        "libcudart-c3a75b33.so.12.8.90",
        "libnvJitLink-0369e686.so.12.8.93",
        "libcublasLt-10b5e663.so.12.8.4.1",
        "libcublas-031ce6c2.so.12.8.4.1",
        "libcusparse-10bf8114.so.12.5.8.93",
        "libcusolver-6e8b369b.so.11.7.3.90",
        "libcudss-1813f5de.so.0.7.1",
    ]:
        try:
            ctypes.CDLL(str(_venv_libs / lib), ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]


def on_load():
    """Called when plugin is loaded."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("Lichtfeld-COLMAP-Plugin plugin loaded")


def on_unload():
    """Called when plugin is unloaded."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("Lichtfeld-COLMAP-Plugin plugin unloaded")
