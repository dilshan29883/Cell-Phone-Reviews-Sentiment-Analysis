@echo off
echo ==========================================
echo   SENTIMENT ANALYSIS PROJECT RUNNER
echo ==========================================
echo.

if "%1"=="" (
    echo Available commands:
    echo   setup    - Install requirements and setup project
    echo   api      - Start web server ^(recommended^)
    echo   demo     - Run quick demo
    echo   notebook - Open Jupyter notebook
    echo   test     - Run system tests
    echo.
    echo Usage: run_project.bat ^<command^>
    echo Example: run_project.bat api
    goto :end
)

if "%1"=="setup" (
    echo Setting up project...
    python run_project.py setup
    goto :end
)

if "%1"=="api" (
    echo Starting web server...
    echo Web interface will be available at: http://127.0.0.1:5000
    echo Press Ctrl+C to stop the server
    echo.
    python run_project.py api
    goto :end
)

if "%1"=="demo" (
    echo Running demo...
    python run_project.py demo
    goto :end
)

if "%1"=="notebook" (
    echo Opening Jupyter notebook...
    python run_project.py notebook
    goto :end
)

if "%1"=="test" (
    echo Running tests...
    python run_project.py test
    goto :end
)

echo Unknown command: %1
echo Use 'run_project.bat' without arguments to see available commands.

:end
pause
