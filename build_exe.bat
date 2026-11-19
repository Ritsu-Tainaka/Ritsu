@echo off
setlocal
cd /d "%~dp0"
title Manga Cevirici - EXE derleme

REM Kullanim:  build_exe.bat          -> normal (konsolsuz) surum
REM            build_exe.bat debug    -> hata gormek icin konsollu surum

set MODE=%1
set CONSOLE=--noconsole
if /I "%MODE%"=="debug" set CONSOLE=--console

REM 1) Python bul (3.10-3.12; sabitlenmis numpy/Pillow 3.13+ icin hazir paket sunmuyor)
set PY=
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY ( py -3.11 -c "" >nul 2>nul && set "PY=py -3.11" )
if not defined PY ( py -3.10 -c "" >nul 2>nul && set "PY=py -3.10" )
if not defined PY ( where python >nul 2>nul && set PY=python )
if not defined PY (
  echo [HATA] Python 3.12 bulunamadi. python.org'dan kurun ve "Add python to PATH" secin.
  pause & exit /b 1
)

REM 2) Sanal ortam + paketler
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Sanal ortam olusturuluyor...
  %PY% -m venv .venv || ( echo [HATA] venv olusturulamadi & pause & exit /b 1 )
)
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" || (
  echo [HATA] .venv desteklenmeyen bir Python surumuyle olusturulmus. .venv klasorunu silip tekrar calistirin.
  pause & exit /b 1
)
echo [2/4] Paketler kuruluyor/guncelleniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller || ( echo [HATA] Paket kurulumu basarisiz & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pip install --no-deps "rapidocr_onnxruntime==1.4.4" || echo [UYARI] RapidOCR kurulamadi; EXE Tesseract ile calisacak.

REM 3) Eski derlemeyi temizle
echo [3/4] Eski derleme temizleniyor...
if exist build rmdir /s /q build
if exist dist\MangaCevirici rmdir /s /q dist\MangaCevirici
if exist MangaCevirici.spec del /q MangaCevirici.spec

REM 4) Derle
echo [4/4] EXE derleniyor (birkac dakika surebilir)...
".venv\Scripts\python.exe" -m PyInstaller %CONSOLE% --onedir --noconfirm --clean ^
  --name MangaCevirici ^
  --add-data "fonts;fonts" ^
  --hidden-import keyring.backends.Windows ^
  --hidden-import pymupdf ^
  --collect-all rapidocr_onnxruntime ^
  --collect-data sv_ttk ^
  --exclude-module networkx ^
  --exclude-module pymupdf4llm ^
  --exclude-module pymupdf_layout ^
  main.py
if errorlevel 1 ( echo [HATA] Derleme basarisiz. Yukaridaki mesaji kontrol edin. & pause & exit /b 1 )

echo.
echo ============================================================
echo  TAMAM:  dist\MangaCevirici\MangaCevirici.exe
echo  Tum "dist\MangaCevirici" klasorunu istediginiz yere tasiyabilirsiniz.
echo  Ekonomik mod icin Tesseract ayrica kurulu olmali (EXE icine gomulu degil).
echo  Acilmazsa:  build_exe.bat debug   ile konsollu surum derleyip hatayi gorun.
echo ============================================================
pause
