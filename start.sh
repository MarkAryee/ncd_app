pip install -r requirements.txt

echo "Starting app on port $PORT..."
uvicorn main:app --reload --host 0.0.0.0 --port $PORT
