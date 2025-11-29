@echo off
chcp 65001 >nul
REM Quick test script for suber_short.mp4 video
REM Just double-click to run!

setlocal enabledelayedexpansion

set "VIDEO_FILE=suber_short.mp4"

if not exist "!VIDEO_FILE!" (
    echo Error: File "!VIDEO_FILE!" not found!
    echo Make sure the video file is in the same folder as this script.
    pause
    exit /b 1
)


echo.
echo Choose mode:
echo   1. Statistics only (fast)
echo   2. Video + Statistics (slow)
echo.
set /p MODE_CHOICE="Enter choice (1 or 2, default is 1): "

if "!MODE_CHOICE!"=="" set "MODE_CHOICE=1"
if "!MODE_CHOICE!"=="2" (
    set "MODE=video"
) else (
    set "MODE=stats"
)

echo.
echo.

if "!MODE!"=="stats" (
    echo Mode: Statistics only
    echo This may take some time...
    echo.
    curl -v -X POST "http://localhost:8000/detect/video" ^
      -F "file=@!VIDEO_FILE!" ^
      -F "return_video=false" ^
      -F "return_statistics=true" > temp_response.txt 2>&1
) else (
    echo Mode: Video + Statistics
    echo This will take a LONG time...
    echo.
    curl -v -X POST "http://localhost:8000/detect/video" ^
      -F "file=@!VIDEO_FILE!" ^
      -F "return_video=true" ^
      -F "return_statistics=true" > temp_response.txt 2>&1
)

type temp_response.txt
echo.

findstr /C:"\"detail\"" temp_response.txt >nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Check the error message above
    del temp_response.txt
    pause
    exit /b 1
)

findstr /C:"\"success\"" temp_response.txt >nul
if %ERRORLEVEL% EQU 0 (
    echo.
) else (
    echo.
)

del temp_response.txt 2>nul

echo.
pause

