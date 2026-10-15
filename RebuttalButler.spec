# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = [('assets/rebuttal_butler_logo.png', 'assets'), ('assets/rebuttal_butler.ico', 'assets')]
binaries = [('.buildenv/Lib/site-packages/nvidia/cublas/bin/*.dll', 'nvidia/cublas/bin'), ('.buildenv/Lib/site-packages/nvidia/cudnn/bin/*.dll', 'nvidia/cudnn/bin')]
datas += collect_data_files('faster_whisper')
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('lameenc')
binaries += collect_dynamic_libs('miniaudio')


a = Analysis(
    ['main_standalone.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['_portaudiowpatch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RebuttalButler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/rebuttal_butler.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RebuttalButler',
)
