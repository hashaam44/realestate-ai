$projectPath = "D:\ML intern prj\Revision ML_DL\ML\RealEstate AI"

Set-Location $projectPath

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectPath'; .\venv\Scripts\Activate.ps1; python -m uvicorn api.main:app --reload"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectPath'; python -m http.server 5500 --directory frontend"

Start-Sleep -Seconds 2

Start-Process "http://127.0.0.1:5500"