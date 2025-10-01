@echo off
setlocal

REM ==============================
REM Download-Quellen
REM ==============================
set URL1=https://github.com/pardonmycode/export_survi/raw/main/Survivor2.3.rar
set FILE1=Survivor2.3.rar
set OUTDIR1=SurvivorExtracted

set URL2=https://github.com/pardonmycode/export_survi/raw/main/endless2.3.zip
set FILE2=endless2.3.zip
set OUTDIR2=EndlessExtracted

REM ==============================
REM Prüfen ob 7z.exe verfügbar ist
REM ==============================
where 7z >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] 7-Zip wurde nicht gefunden.
    echo     Es wird nun heruntergeladen und entpackt.
    
    set ZIPURL=https://www.7-zip.org/a/7zr.exe
    set ZIPFILE=7zr.exe
    
    powershell -Command "Invoke-WebRequest '%ZIPURL%' -OutFile '%ZIPFILE%'"
    
    if not exist "%ZIPFILE%" (
        echo Fehler: Konnte 7-Zip nicht herunterladen.
        exit /b 1
    )
    
    echo 7-Zip (7zr.exe) erfolgreich heruntergeladen.
    set PATH=%CD%;%PATH%
    set SEVENZIP=7zr.exe
) else (
    set SEVENZIP=7z.exe
)

REM ==============================
echo === Lade Survivor2.3.rar herunter ===
powershell -Command "Invoke-WebRequest '%URL1%' -OutFile '%FILE1%'"

if not exist "%FILE1%" (
    echo Fehler: %FILE1% konnte nicht heruntergeladen werden.
    exit /b 1
)

echo === Lade endless2.3.zip herunter ===
powershell -Command "Invoke-WebRequest '%URL2%' -OutFile '%FILE2%'"

if not exist "%FILE2%" (
    echo Fehler: %FILE2% konnte nicht heruntergeladen werden.
    exit /b 1
)

REM ==============================
echo === Entpacke Survivor2.3.rar ===
%SEVENZIP% x "%FILE1%" -o"%OUTDIR1%" -y
if errorlevel 1 (
    echo Fehler beim Entpacken von %FILE1%.
    exit /b 1
)

echo === Entpacke endless2.3.zip ===
%SEVENZIP% x "%FILE2%" -o"%OUTDIR2%" -y
if errorlevel 1 (
    echo Fehler beim Entpacken von %FILE2%.
    exit /b 1
)

echo === Alle Dateien erfolgreich entpackt! ===
echo - Survivor liegt in %OUTDIR1%
echo - Endless liegt in %OUTDIR2%

endlocal
pause
