from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from datetime import datetime
import base64
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
            body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; text-align: center; }
            video, canvas { width: 400px; height: 300px; border: 2px solid #ccc; margin: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>Attendance System</h1>
        <video id="video" autoplay></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <br>
        <button onclick="startCamera()">Start Camera</button>
        <button onclick="capturePhoto()">Mark Attendance</button>
        <div id="result"></div>

        <script>
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let ctx = canvas.getContext('2d');

            function startCamera() {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                        video.srcObject = stream;
                    })
                    .catch(err => {
                        document.getElementById('result').innerHTML = '<p style="color:red;">Camera access denied</p>';
                    });
            }

            function capturePhoto() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                let imageData = canvas.toDataURL('image/jpeg');
                
                fetch('/process/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                })
                .then(response => response.text())
                .then(data => {
                    document.getElementById('result').innerHTML = data;
                });
            }
        </script>
    </body>
    </html>
    """

@app.post("/process/")
async def process_image(request: Request):
    try:
        data = await request.json()
        image_data = data['image'].split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return f'<p style="color:green;">✅ Attendance Marked at {timestamp}</p>'
        else:
            return '<p style="color:red;">❌ No face detected</p>'
            
    except Exception as e:
        return '<p style="color:red;">Error processing image</p>'
