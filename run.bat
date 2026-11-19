@echo off
setlocal
cd /d "%~dp0"

REM Python 3.10-3.12 gerekli (sabitlenmis numpy/Pillow surumlerinin 3.13+ icin hazir paketi yok)
set PY=
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY ( py -3.11 -c "" >nul 2>nul && set "PY=py -3.11" )
if not defined PY ( py -3.10 -c "" >nul 2>nul && set "PY=py -3.10" )
if not defined PY ( where python >nul 2>nul && set PY=python )
if not defined PY (
  echo [HATA] Python 3.12 bulunamadi. python.org'dan Python 3.12 kurun ve "Add python to PATH" secin.
  pause & exit /b 1
)

REM Kurulum yarida kaldiysa bir sonraki acilista yeniden denenir (isaret dosyasi)
REM v3: yeni paketler (RapidOCR, onnxruntime, sv-ttk tema) eklendi; eski kurulumlar bir kez guncellenir
if not exist ".venv\kurulum_v3.txt" (
  if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv || ( echo [HATA] Sanal ortam olusturulamadi & pause & exit /b 1 )
  )
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" || (
    echo [HATA] .venv desteklenmeyen bir Python surumuyle olusturulmus. .venv klasorunu silin, Python 3.12 kurun ve tekrar calistirin.
    pause & exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || ( echo [HATA] Paket kurulumu basarisiz. Internet baglantisini kontrol edip tekrar calistirin. & pause & exit /b 1 )
  ".venv\Scripts\python.exe" -m pip install --no-deps "rapidocr_onnxruntime==1.4.4" || echo [UYARI] RapidOCR kurulamadi; Ekonomik modda Tesseract kullanilacak.
  echo ok> ".venv\kurulum_v3.txt"
)
".venv\Scripts\python.exe" main.py
pause
