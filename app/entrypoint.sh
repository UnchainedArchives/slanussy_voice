# Initialize models
python3 -c "from main import download_default_model; download_default_model()"

# Start the server
exec /opt/bark_env/bin/uvicorn main:app --host 0.0.0.0 --port 5001