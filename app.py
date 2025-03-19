import os
import subprocess
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import streamlit as st
import threading

app = FastAPI()

# Serve static files (if needed for UI assets)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for serving the chatbot iframe
@app.get("/", response_class=HTMLResponse)
async def get_chatbot():
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Chatbot</title>
        </head>
        <body>
            <h1>Welcome to the AI Chatbot</h1>
            <iframe src="http://localhost:8501" width="100%" height="600" frameborder="0"></iframe>
        </body>
        </html>
    """


# Function to start Streamlit
def start_streamlit():
    subprocess.run(["streamlit", "run", "bot1.py"])

# Start Streamlit in a separate thread
thread = threading.Thread(target=start_streamlit)
thread.start()