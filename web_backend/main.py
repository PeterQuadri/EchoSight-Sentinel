import os
import io
import json
import base64
import asyncio
import numpy as np
import torch
import librosa
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from collections import deque, Counter
from datetime import datetime
from emergency_video_llm import EmergencyVideoAnalyzer
from whatsapp_notifier import WhatsAppNotifier
from twilio.twiml.messaging_response import MessagingResponse
import torch.nn as nn
import threading

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Components
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
SANDBOX_NUM = os.getenv("TWILIO_SANDBOX_NUMBER")
USER_PHONE = os.getenv("USER_PHONE_NUMBER")

# IMPORTANT: Headless=True prevents the backend from locking the laptop camera
video_analyzer = EmergencyVideoAnalyzer(groq_api_key=GROQ_API_KEY, headless=True)
whatsapp_notifier = WhatsAppNotifier(TWILIO_SID, TWILIO_TOKEN, SANDBOX_NUM, USER_PHONE)

# High-performance CNN architecture (Matches training script)
class EmergencySoundCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(EmergencySoundCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout_rate * 0.5)
        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x

# Load Audio Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "../models/best_model.pth"
CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams']

audio_model = EmergencySoundCNN(num_classes=len(CLASS_NAMES)).to(device)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        audio_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        audio_model.load_state_dict(checkpoint)
    audio_model.eval()
    print(f"✅ Audio AI Model loaded successfully.")
else:
    print(f"❌ Audio AI Model NOT FOUND at {MODEL_PATH}")

def preprocess_audio_array(audio_array, sr):
    try:
        # Aggressive Noise Floor (Matching script)
        rms = np.sqrt(np.mean(audio_array**2))
        if rms < 0.002: # Lower threshold to match script
             return torch.zeros((1, 1, 128, 44)).to(device) # Return silent tensor
            
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array, sr=sr, n_mels=128, n_fft=1024, hop_length=512, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Improved normalization matching script
        db_range = mel_spec_db.max() - mel_spec_db.min()
        if db_range > 10:
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (db_range + 1e-8)
        else:
            # If very quiet, just scale relative to a "safe" floor
            mel_spec_db = (mel_spec_db + 80) / 80
            mel_spec_db = np.clip(mel_spec_db, 0, 1)
            
        return torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error preprocessing audio array: {e}")
        return None

async def verify_and_notify(class_name, confidence, websocket: WebSocket):
    print(f"🧐 Visually verifying {class_name}...")
    try:
        if websocket:
            await websocket.send_json({
                "type": "alert_verifying",
                "class": class_name
            })
            
        # 1. Vision Verification with Groq (Run in thread to avoid blocking event loop)
        loop = asyncio.get_running_loop()
        vision_res = await loop.run_in_executor(None, video_analyzer.verify_alert, class_name)
        
        if websocket:
            await websocket.send_json({
                "type": "vision_verification",
                "decision": vision_res.get("decision"),
                "reasoning": vision_res.get("reasoning")
            })

        if vision_res.get("decision") == "True Emergency":
            # 2. WhatsApp Notification
            whatsapp_notifier.send_alert(
                class_name, 
                confidence, 
                vision_res.get("reasoning"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
    except Exception as e:
        print(f"Error in verification task: {e}")

@app.get("/")
async def root():
    return {"status": "EchoSight Backend Online", "health": "OK"}

@app.get("/health")
def health():
    return {"status": "online"}

@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    """Handle incoming WhatsApp messages (Merged from whatsapp_server)"""
    incoming_msg = Body.lower().strip()
    print(f"📥 Received WhatsApp from {From}: {incoming_msg}")
    
    resp = MessagingResponse()
    status_keywords = ["status", "check", "how", "situation", "house", "monitoring", "update", "happening"]
    
    if any(kw in incoming_msg for kw in status_keywords):
        print("👁️ TRIGGER: Remote Status Check requested.")
        
        # Start the verification task as a background task
        def perform_check():
            try:
                result = video_analyzer.verify_alert(audio_class=None)
                reasoning_text = result.get('reasoning', "I can see the room, but I couldn't generate a detailed description right now.")
                report = f"🏠 *Home Status Update*\n\n{reasoning_text}"
                whatsapp_notifier.send_message(report)
            except Exception as e:
                print(f"❌ ERROR: Status check thread failed: {e}")
        
        threading.Thread(target=perform_check, daemon=True).start()
        resp.message("👁️ One moment, I'm checking the live feed for you...")
    else:
        reply = "🤖 *EchoSight Sentinel Bot*\n\nI didn't quite catch that. Send 'status', 'check', or 'how is the house' to get a real-time visual update!"
        resp.message(reply)
        
    return Response(content=str(resp), media_type="application/xml")

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Client Connected to Stream Socket")
    
    SR = 22050
    CHUNK_DURATION = 3 
    BUFFER_SIZE = SR * CHUNK_DURATION 
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    pred_history = deque(maxlen=2)
    last_alert_times = {}
    COOLDOWN_SECONDS = 5
    
    try:
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)
            
            img_b64 = packet.get("image")
            if img_b64:
                video_analyzer.add_frame(img_b64)

            audio_b64 = packet.get("audio")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                if len(chunk) > 0:
                    audio_buffer = np.roll(audio_buffer, -len(chunk))
                    audio_buffer[-len(chunk):] = chunk
                
                if np.max(np.abs(audio_buffer)) > 0:
                    input_tensor = preprocess_audio_array(audio_buffer, SR)
                    
                    if input_tensor is not None:
                        with torch.no_grad():
                            outputs = audio_model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                            
                            current_preds = {
                                CLASS_NAMES[i]: float(probs[i]) 
                                for i in range(len(CLASS_NAMES))
                            }
                            
                            conf, pred_idx = torch.max(probs, 0)
                            pred_class = CLASS_NAMES[pred_idx.item()]
                            conf_val = conf.item()
                            pred_history.append((pred_class, conf_val))
                            
                            if len(pred_history) >= 2:
                                recent_classes = [p[0] for p in pred_history]
                                counts = Counter(recent_classes)
                                most_common, freq = counts.most_common(1)[0]
                                
                                if freq >= 2 and most_common != 'background' and most_common == pred_class:
                                    avg_conf = sum(p[1] for p in pred_history if p[0] == most_common) / freq
                                    
                                    if avg_conf > 0.80:
                                        now = asyncio.get_running_loop().time()
                                        last_time = last_alert_times.get(most_common, 0)
                                        
                                        if now - last_time > COOLDOWN_SECONDS:
                                            print(f"🚨 ROBUST ALERT: {most_common} ({avg_conf*100:.1f}%)")
                                            last_alert_times[most_common] = now
                                            asyncio.create_task(verify_and_notify(most_common, avg_conf, websocket))
                                            pred_history.clear()
                                            audio_buffer.fill(0)

                            await websocket.send_json({
                                "type": "audio_analysis",
                                "predictions": current_preds,
                                "smoothed_predictions": current_preds
                            })

    except WebSocketDisconnect:
        print("❌ Client Disconnected")

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for Render (defaults to 8000 for local)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
