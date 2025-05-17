chmod +x start.sh
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
