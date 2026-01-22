"""
Integrated Emergency Detection System
Combines audio detection with real-time video analysis using LLAMA Vision
"""

import cv2
import base64
import json
from groq import Groq
from datetime import datetime
from typing import Dict
import time
import threading
import queue
import numpy as np


class VideoAnalyzer:
    """
    Real-time video analyzer using LLAMA vision models via Groq
    Analyzes live feed when triggered by audio alerts
    """
    
    def __init__(self, groq_api_key: str, video_source=0, model: str = "llama-3.2-90b-vision-preview"):
        """
        Initialize the video analyzer
        
        Args:
            groq_api_key: Your Groq API key
            video_source: Camera index (0 for default webcam) or video stream URL
            model: LLAMA vision model to use
        """
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.video_source = video_source
        
        # Video capture
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_running = False
        
        # Alert queue for audio system integration
        self.alert_queue = queue.Queue()
        
        # Analysis thread
        self.analysis_thread = None
        self.analysis_running = False
        
    def start_video_capture(self):
        """Start continuous video capture in background thread"""
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video source: {self.video_source}")
        
        self.capture_running = True
        
        def capture_loop():
            """Continuously capture frames from camera"""
            while self.capture_running:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frame = frame
                time.sleep(0.03)  # ~30 FPS
        
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
        print("✅ Video capture started")
    
    def stop_video_capture(self):
        """Stop video capture"""
        self.capture_running = False
        if self.cap:
            self.cap.release()
        print("🛑 Video capture stopped")
    
    def encode_frame(self, frame) -> str:
        """Encode video frame to base64"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def create_analysis_prompt(self, audio_alert: str) -> str:
        """Create the prompt for LLAMA vision model"""
        prompt = f"""You are an emergency detection system analyzing a video frame after an audio alert was triggered.

AUDIO ALERT DETECTED: {audio_alert}

Analyze this image carefully for signs of a genuine emergency. Look for:

EMERGENCY INDICATORS:
- People running, fleeing, or moving erratically
- Individuals hiding under tables, desks, or behind objects
- Panic-like body language (crouching, covering head, defensive postures)
- Crowd dispersal or people moving toward exits
- Visible fear responses or distress
- Any signs of danger or threat

FALSE POSITIVE INDICATORS:
- TV, monitor, or screen visible showing media content
- People watching a screen calmly
- Normal, calm behavior and posture
- No signs of distress or emergency response
- Staged or entertainment setting

Respond ONLY with a valid JSON object in this exact format:
{{
    "decision": "True Emergency" or "False Positive",
    "confidence": 0.0 to 1.0,
    "visual_indicators": ["list", "of", "observed", "indicators"],
    "reasoning": "brief explanation of decision",
    "people_detected": number,
    "behavior_assessment": "description of observed behaviors"
}}

Be precise and objective. Lives may depend on this analysis."""
        
        return prompt
    
    def analyze_frame(self, frame, audio_alert: str) -> Dict:
        """Analyze a single frame using LLAMA vision model"""
        try:
            base64_image = self.encode_frame(frame)
            prompt = self.create_analysis_prompt(audio_alert)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            analysis = json.loads(response_text)
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['audio_alert'] = audio_alert
            
            return analysis
            
        except Exception as e:
            print(f"❌ Video analysis error: {e}")
            return {
                "decision": "Error",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_alert(self, audio_alert: str) -> Dict:
        """
        THIS IS THE MAIN FUNCTION YOUR AUDIO SYSTEM CALLS
        Captures and analyzes multiple frames when audio alert is triggered
        
        Args:
            audio_alert: Type of audio alert (gun_shots, screams, glass_breaking, etc.)
            
        Returns:
            Analysis result with final decision
        """
        print(f"\n{'='*70}")
        print(f"🚨 AUDIO ALERT RECEIVED: {audio_alert}")
        print(f"{'='*70}")
        print("📹 Capturing frames for video analysis...")
        
        # Capture multiple frames for robust analysis
        frames = []
        for i in range(3):
            with self.frame_lock:
                if self.latest_frame is not None:
                    frames.append(self.latest_frame.copy())
            time.sleep(0.3)  # 300ms between captures
        
        if not frames:
            print("❌ No video frames available")
            return {
                "final_decision": "Error",
                "error": "No video frames available",
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"✅ Captured {len(frames)} frames")
        print("🤖 Analyzing with LLAMA Vision Model...")
        
        # Analyze each frame
        analyses = []
        for i, frame in enumerate(frames):
            print(f"  Analyzing frame {i+1}/{len(frames)}...", end='\r')
            analysis = self.analyze_frame(frame, audio_alert)
            analyses.append(analysis)
            time.sleep(0.5)  # Rate limiting
        
        print()  # New line after analysis
        
        # Aggregate results (majority voting)
        emergency_votes = sum(1 for a in analyses if a.get('decision') == 'True Emergency')
        avg_confidence = sum(a.get('confidence', 0) for a in analyses) / len(analyses)
        
        final_decision = "True Emergency" if emergency_votes >= 2 else "False Positive"
        
        result = {
            "final_decision": final_decision,
            "confidence": avg_confidence,
            "emergency_votes": emergency_votes,
            "total_frames": len(analyses),
            "audio_alert": audio_alert,
            "timestamp": datetime.now().isoformat(),
            "individual_analyses": analyses
        }
        
        # Print result
        print(f"\n{'='*70}")
        print("📊 VIDEO ANALYSIS RESULT")
        print(f"{'='*70}")
        print(f"Decision: {final_decision}")
        print(f"Confidence: {avg_confidence*100:.2f}%")
        print(f"Emergency Votes: {emergency_votes}/{len(analyses)}")
        print(f"{'='*70}\n")
        
        return result
    
    def start_analysis_loop(self):
        """Start background thread to process alert queue"""
        self.analysis_running = True
        
        def analysis_loop():
            """Process video analysis requests from queue"""
            while self.analysis_running:
                try:
                    audio_alert = self.alert_queue.get(timeout=1)
                    result = self.analyze_alert(audio_alert)
                    
                    # Take action based on result
                    if result['final_decision'] == 'True Emergency':
                        print("🚨 CONFIRMED EMERGENCY - SENDING ALERTS")
                        # TODO: Add WhatsApp/SMS notification here
                        # self.send_emergency_notification(result)
                    else:
                        print("✅ False positive - No action needed")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Analysis loop error: {e}")
        
        self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("✅ Video analysis loop started")
    
    def stop_analysis_loop(self):
        """Stop analysis loop"""
        self.analysis_running = False
    
    def on_audio_alert(self, alert_type: str):
        """
        CALLBACK FOR AUDIO SYSTEM
        Queue alert for video analysis
        
        Args:
            alert_type: Type of audio alert from your audio detection system
        """
        self.alert_queue.put(alert_type)


# ============================================================
# INTEGRATION WRAPPER - MODIFY YOUR AUDIO SYSTEM
# ============================================================

class IntegratedEmergencySystem:
    """
    Wrapper to integrate video analyzer with your audio detection system
    """
    
    def __init__(self, groq_api_key: str, video_source=0):
        """
        Initialize integrated system
        
        Args:
            groq_api_key: Your Groq API key
            video_source: Camera index or video stream URL
        """
        self.video_analyzer = VideoAnalyzer(
            groq_api_key=groq_api_key,
            video_source=video_source,
            model="llama-3.2-90b-vision-preview"  # or "llama-3.2-11b-vision-preview" for faster
        )
    
    def start(self):
        """Start the integrated system"""
        print("\n" + "="*70)
        print("🚀 STARTING INTEGRATED EMERGENCY DETECTION SYSTEM")
        print("="*70)
        print("Audio Detection: ✅ (from your system)")
        print("Video Analysis: Starting...")
        
        # Start video capture
        self.video_analyzer.start_video_capture()
        
        # Start analysis loop
        self.video_analyzer.start_analysis_loop()
        
        print("\n✅ System ready!")
        print("Waiting for audio alerts to trigger video analysis...\n")
    
    def stop(self):
        """Stop the integrated system"""
        print("\n🛑 Stopping integrated system...")
        self.video_analyzer.stop_analysis_loop()
        self.video_analyzer.stop_video_capture()
        print("✅ System stopped")
    
    def on_audio_alert(self, alert_type: str):
        """
        CALL THIS FROM YOUR AUDIO DETECTION SYSTEM
        
        This is the main integration point!
        When your audio system detects something, call this method.
        
        Args:
            alert_type: Type of audio alert (gun_shots, screams, glass_breaking, etc.)
        """
        self.video_analyzer.on_audio_alert(alert_type)


# ============================================================
# HOW TO INTEGRATE WITH YOUR AUDIO SYSTEM
# ============================================================

"""
INTEGRATION INSTRUCTIONS:

1. In your realtime_detection_system.py, add this at the top:
   
   from integrated_system import IntegratedEmergencySystem
   
2. In the RealTimeDetector.__init__() method, add:
   
   self.video_system = None  # Will be set when starting
   
3. In the RealTimeDetector.start() method, add BEFORE starting audio:
   
   # Initialize video analysis system
   GROQ_API_KEY = "your_groq_api_key_here"
   self.video_system = IntegratedEmergencySystem(
       groq_api_key=GROQ_API_KEY,
       video_source=0  # 0 for default webcam
   )
   self.video_system.start()
   
4. In the RealTimeDetector.trigger_alert() method, add BEFORE or AFTER logging:
   
   # Trigger video analysis
   if self.video_system:
       self.video_system.on_audio_alert(class_name)
   
5. That's it! Now when audio detects an emergency, video analysis automatically runs!

"""


# ============================================================
# STANDALONE TESTING
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("INTEGRATED EMERGENCY DETECTION SYSTEM - STANDALONE TEST")
    print("="*70)
    
    # Initialize system
    GROQ_API_KEY = "your_groq_api_key_here"  # REPLACE WITH YOUR KEY
    
    system = IntegratedEmergencySystem(
        groq_api_key=GROQ_API_KEY,
        video_source=0
    )
    
    # Start system
    system.start()
    
    # Simulate audio alerts for testing
    print("\n" + "="*70)
    print("TESTING MODE - Simulating audio alerts")
    print("="*70)
    print("Press Enter to simulate alerts, 'q' to quit\n")
    
    test_alerts = ["gun_shots", "screams", "glass_breaking"]
    alert_index = 0
    
    try:
        while True:
            user_input = input(f"Press Enter to test '{test_alerts[alert_index]}' alert (or 'q' to quit): ")
            
            if user_input.lower() == 'q':
                break
            
            # Simulate audio alert
            print(f"\n🎵 Simulating audio detection: {test_alerts[alert_index]}")
            system.on_audio_alert(test_alerts[alert_index])
            
            # Wait for analysis to complete
            time.sleep(8)
            
            alert_index = (alert_index + 1) % len(test_alerts)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        system.stop()
        print("\n✅ Test complete")
