@echo off
setlocal

REM #project folder
cd /d "C:\Users\Matthew\OneDrive\Desktop\machine learning- baseball"

REM #optional: activate conda env
REM #change "baseball-env" to your actual environment name
call conda activate baseball-env

REM #or if you use venv instead, comment the conda line above and use this:
REM call .venv\Scripts\activate

REM #make sure git lfs is available
git lfs install

REM #sync first
git pull origin main

REM #update current-year statcast data
python update_statcast_ytd.py
if errorlevel 1 goto :error

REM #retrain models
python retrain_models.py
if errorlevel 1 goto :error

REM #stage everything
git add .

REM #commit only if there are changes
git diff --cached --quiet
if %errorlevel%==0 goto :nochanges

git commit -m "Auto-update Statcast data and retrain models"
git push origin main
if errorlevel 1 goto :error

echo.
echo SUCCESS: update, retrain, and push completed.
goto :end

:nochanges
echo.
echo No changes detected. Nothing to commit.
goto :end

:error
echo.
echo ERROR: pipeline failed. Check terminal/log output.
exit /b 1

:end
endlocal
exit /b 0