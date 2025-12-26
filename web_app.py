from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from datetime import datetime
import io
from PIL import Image

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attendance System</title>
        <style>
            body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            input[type="file"] { margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Attendance System</h1>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <div class="upload-area">
                <p>Upload an image to mark attendance</p>
                <input name="file" type="file" accept="image/*" required>
                <br><button type="submit">Mark Attendance</button>
            </div>
        </form>
    </body>
    </html>
    """

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return HTMLResponse(f"""
            <html><body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
                <h2>✅ Attendance Marked!</h2>
                <p>Face detected at {timestamp}</p>
                <a href="/">← Back</a>
            </body></html>
            """)
        else:
            return HTMLResponse(f"""
            <html><body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
                <h2>❌ No Face Detected</h2>
                <p>Please upload a clear image with a visible face</p>
                <a href="/">← Try Again</a>
            </body></html>
            """)
    except Exception as e:
        return HTMLResponse(f"<html><body><h2>Error processing image</h2><a href='/'>← Back</a></body></html>")
