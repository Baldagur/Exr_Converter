# PyInstaller spec: one folder with a double-clickable Linux executable with the K1S1 LUT, OCIO and ffmpeg inside.
# Build with packaging/build_linux.sh (needs internet once, on the build machine only).
import glob
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

root = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821

datas = [(os.path.join(root, "exr_converter", "resources"), "resources")]
binaries = []
ffmpeg = os.path.join(root, "build_deps", "ffmpeg")
if os.path.isfile(ffmpeg):
    binaries.append((ffmpeg, "bin"))
else:
    print("WARNING: build_deps/ffmpeg not found - QuickTime export will need ffmpeg on PATH")

# Qt 6.5+ needs libxcb-cursor for its X11 plugin; Rocky Linux only has it in EPEL, so bundle it.
for lib in glob.glob("/usr/lib64/libxcb-cursor.so.0*") + glob.glob("/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0*"):
    binaries.append((lib, "."))

binaries += collect_dynamic_libs("PyOpenColorIO") + collect_dynamic_libs("OpenEXR")

a = Analysis(  # noqa: F821
    [os.path.join(root, "run.py")],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=["PyOpenColorIO", "OpenEXR", "PIL.JpegImagePlugin", "PIL.PngImagePlugin"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
# onedir (not onefile): nothing is unpacked to /tmp at launch, so it starts fast and works when
# /tmp is mounted noexec (common on hardened Rocky installs).
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ExrConverter",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ExrConverter",
)
