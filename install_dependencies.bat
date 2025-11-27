@echo off
echo Installing all dependencies for the project...
echo.

.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Installation complete!
echo.
pause

