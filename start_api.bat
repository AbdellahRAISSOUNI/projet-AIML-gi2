@echo off
echo Starting FastAPI server...
python -m uvicorn api:app --host 0.0.0.0 --port 8000
pause 