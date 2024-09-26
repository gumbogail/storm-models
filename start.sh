#!/bin/bash
# Activate virtual environment (if applicable)
source venv/bin/activate
# Run your application
python main.py
chmod +x start.sh
uvicorn main:app --host 0.0.0.0 --port 8080