::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Root directory of this script (no trailing backslash)
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM Timestamp for package name
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TS=%%i"

set "DIST=%ROOT%\dist"
if not exist "%DIST%" mkdir "%DIST%"

set "PKG_NAME=midimaster_%TS%"
set "STAGE=%DIST%\%PKG_NAME%"

echo Staging to: "%STAGE%"
if exist "%STAGE%" rmdir /S /Q "%STAGE%"
mkdir "%STAGE%"

REM Directories to include if present
set "DIRS=analyzers audio configs docs examples generators genres gui projects structures"

REM Directory exclusions (by name, applied within each included directory)
set "EXCLUDE_DIRS=.git .roo __pycache__ .pytest_cache output test_outputs test_split_output test_split_output2 test_split_output3 pop_a_dorian_parts .venv venv env .mypy_cache .ruff_cache .vscode"

REM Global file patterns to exclude
set "XF=*.pyc *.pyo *.pyd *.so *.dll *.dylib *.log *.tmp *.cache *.mid *.html *.csv *.tsv *.png *.jpg *.jpeg *.gif *.svg"

REM Robocopy flags
set "ROBOFLAGS=/E /NFL /NDL /NJH /NJS /NP"
set "ROBOX=/XO /XF %XF%"

for %%D in (%DIRS%) do (
  if exist "%ROOT%\%%D" (
    echo.
    echo Copying directory: %%D
    set "SRC=%ROOT%\%%D"
    set "DST=%STAGE%\%%D"
    mkdir "!DST!" >nul 2>&1

    set "XD_SWITCHES="
    for %%X in (%EXCLUDE_DIRS%) do (
      set "XD_SWITCHES=!XD_SWITCHES! /XD ""!SRC!\%%X"""
    )

    robocopy "!SRC!" "!DST!" %ROBOFLAGS% %ROBOX% !XD_SWITCHES!
    set "RC=!ERRORLEVEL!"
    if !RC! GEQ 8 (
      echo robocopy failed for %%D with code !RC!
      exit /b !RC!
    )
  ) else (
    echo Skipping missing directory: %%D
  )
)

echo.
echo Copying top-level files...
robocopy "%ROOT%" "%STAGE%" *.py README.md requirements.txt LICENSE MIDI_VISUALIZER_README.md ^
  /LEV:1 /NFL /NDL /NJH /NJS /NP ^
  /XF test_*.py *_test.py debug_*.py automated_tests.py direct_test.py simple_test.py detailed_analysis.py fixed_midi_analysis.py comprehensive_midi_analysis.py inspect_*.py output_*.py *.mid *.html *.csv *.tsv

set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo robocopy top-level failed with code %RC%
  exit /b %RC%
)

echo.
echo Creating archive...
powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%\*' -DestinationPath '%DIST%\%PKG_NAME%.zip' -Force"
if errorlevel 1 (
  echo Compress-Archive failed
  exit /b 1
)

echo.
echo Package created: %DIST%\%PKG_NAME%.zip

if /I "%~1"=="CLEAN" (
  echo Removing staging folder...
  rmdir /S /Q "%STAGE%"
)

echo Done.
endlocal
exit /b 0
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::