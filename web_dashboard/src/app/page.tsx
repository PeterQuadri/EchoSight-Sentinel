"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";

export default function Dashboard() {
  const SR = 22050;
  const [analyzing, setAnalyzing] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedAudio, setSelectedAudio] = useState<string>("");
  const [predictions, setPredictions] = useState<Record<string, number>>({
    background: 100,
    glass_breaking: 0,
    gun_shots: 0,
    screams: 0,
  });
  const [verifying, setVerifying] = useState(false);
  const [emergency, setEmergency] = useState(false);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const visualizerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const alertTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const getDevices = async () => {
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const audioDevs = devs.filter(d => d.kind === 'audioinput');
        // Remove the first one (often Stereo Mix / System Audio) as requested
        const filteredDevs = audioDevs.slice(1);
        setDevices(filteredDevs);
        if (filteredDevs.length > 0) setSelectedAudio(filteredDevs[0].deviceId);
      } catch (e) { console.error("Error listing devices:", e); }
    };
    getDevices();
  }, []);

  useEffect(() => {
    if (!analyser || !visualizerCanvasRef.current) return;

    const canvas = visualizerCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "#0a0f1a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add a subtle grid
      ctx.strokeStyle = "rgba(0, 210, 255, 0.05)";
      ctx.beginPath();
      for (let i = 0; i < canvas.width; i += 40) {
        ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height);
      }
      ctx.stroke();

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#00d2ff";
      ctx.shadowBlur = 8;
      ctx.shadowColor = "#00d2ff";
      ctx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const offset = (v - 1.0) * 3;
        const y = ((1.0 + offset) * canvas.height) / 2;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);

        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    draw();
    return () => cancelAnimationFrame(animationId);
  }, [analyser]);

  const startAnalysis = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("❌ Your browser does not support camera/microphone access in this context.");
      return;
    }

    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            deviceId: selectedAudio ? { exact: selectedAudio } : undefined,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
      } catch (audioErr) {
        console.error("Audio capture failed:", audioErr);
        throw new Error("AUDIO_LOCKED");
      }

      const audioTrack = stream.getAudioTracks()[0];
      console.log("🎤 Using Audio Device:", audioTrack.label);

      try {
        const videoStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { max: 15 } }
        });
        videoStream.getVideoTracks().forEach(track => stream.addTrack(track));
      } catch (videoErr) {
        console.warn("Video capture failed - proceeding with AUDIO ONLY");
      }

      if (videoRef.current) videoRef.current.srcObject = stream;

      let wsUrl = process.env.NEXT_PUBLIC_WEBSOCKET_URL || "ws://localhost:8000/ws/stream";

      // Auto-upgrade to wss if on https and user forgot to specify
      if (window.location.protocol === 'https:' && wsUrl.startsWith('ws://')) {
        wsUrl = wsUrl.replace('ws://', 'wss://');
      }

      // Robustness: Ensure path is correct
      if (!wsUrl.includes('/ws/stream')) {
        if (wsUrl.endsWith('/')) wsUrl = wsUrl.slice(0, -1);
        wsUrl += '/ws/stream';
      }

      console.log("🔌 Connecting to:", wsUrl);
      socketRef.current = new WebSocket(wsUrl);

      socketRef.current.onopen = () => {
        console.log("✅ WebSocket Connected");
      };

      socketRef.current.onerror = (err) => {
        console.error("❌ WebSocket Error:", err);
      };

      socketRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "audio_analysis") {
          setPredictions(data.smoothed_predictions || data.predictions);
        } else if (data.type === "vision_verification") {
          if (data.decision === "True Emergency") {
            setEmergency(true);
            setVerifying(false);
            if (alertTimeoutRef.current) clearTimeout(alertTimeoutRef.current);
            alertTimeoutRef.current = setTimeout(() => setEmergency(false), 10000);
          } else {
            setVerifying(false);
            setEmergency(false);
          }
        } else if (data.type === "alert_verifying") {
          setVerifying(true);
        }
      };

      audioContextRef.current = new AudioContext({ sampleRate: SR });

      // Crucial: Auto-resume if suspended (browser security)
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      const source = audioContextRef.current.createMediaStreamSource(stream);
      const newAnalyser = audioContextRef.current.createAnalyser();
      newAnalyser.fftSize = 256;
      setAnalyser(newAnalyser);

      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      source.connect(newAnalyser);
      newAnalyser.connect(processor);
      processor.connect(audioContextRef.current.destination);

      processor.onaudioprocess = (e) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
          }

          let imageB64 = null;
          if (canvasRef.current && videoRef.current) {
            const context = canvasRef.current.getContext("2d");
            context?.drawImage(videoRef.current, 0, 0, 320, 240);
            imageB64 = canvasRef.current.toDataURL("image/jpeg", 0.5).split(",")[1];
          }

          const packet = {
            audio: btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer))),
            image: imageB64
          };
          socketRef.current.send(JSON.stringify(packet));
        }
      };

      setAnalyzing(true);
    } catch (err: any) {
      console.error("Error starting analysis:", err);
      alert("❌ Error: " + err.message);
    }
  };

  const stopAnalysis = () => {
    setAnalyzing(false);
    setEmergency(false);
    if (alertTimeoutRef.current) clearTimeout(alertTimeoutRef.current);
    socketRef.current?.close();
    audioContextRef.current?.close();
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
    }
    setAnalyser(null);
  };

  return (
    <main className="min-h-screen p-8 flex flex-col items-center bg-[#0a0f1a] text-white">
      <div className="w-full max-w-6xl flex justify-between items-center mb-12">
        <div>
          <h1 className="text-4xl font-bold glow-text tracking-tighter">ECHOSIGHT SENTINEL</h1>
          <p className="text-cyan-400 font-mono text-sm">SECURED // WEB_DASHBOARD_V1</p>
        </div>
        <div className="flex items-center gap-4">
          <select
            value={selectedAudio}
            onChange={(e) => setSelectedAudio(e.target.value)}
            disabled={analyzing}
            className="bg-[#1a202c] text-xs border border-white/10 rounded px-2 py-2 outline-none focus:border-cyan-500/50"
          >
            {devices.map(d => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Microphone ${d.deviceId.slice(0, 5)}`}
              </option>
            ))}
          </select>
          <button
            onClick={analyzing ? stopAnalysis : startAnalysis}
            className={`px-8 py-3 rounded-full font-bold transition-all ${analyzing
              ? "bg-red-500/20 text-red-500 border border-red-500 hover:bg-red-500"
              : "bg-cyan-500 text-black hover:bg-cyan-400"
              }`}
          >
            {analyzing ? "STOP SYSTEM" : "INITIALIZE SENTINEL"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full max-w-6xl">
        <div className="lg:col-span-2 glass-panel p-4 relative overflow-hidden flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold flex items-center">
              <span className="w-2 h-2 bg-red-500 rounded-full mr-2 animate-pulse"></span>
              LIVE_SIGHT_FEED
            </h2>
            <span className="text-xs font-mono text-gray-400 uppercase">
              {analyzing ? "Streaming @ 22kHz" : "System Standby"}
            </span>
          </div>
          <div className="relative aspect-video bg-black/40 rounded-lg overflow-hidden border border-white/5 mb-4">
            <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
            <canvas ref={canvasRef} width="320" height="240" className="hidden" />
            {emergency && (
              <div className="absolute inset-0 border-4 border-red-500 pulse-emergency flex items-center justify-center pointer-events-none z-20">
                <div className="bg-red-600 text-white px-6 py-2 rounded-full font-black text-xl animate-bounce text-center">
                  EMERGENCY CONFIRMED<br />
                  <span className="text-sm font-normal">Visual Verification Verified</span>
                </div>
              </div>
            )}
            {verifying && (
              <div className="absolute inset-0 border-4 border-yellow-500 flex items-center justify-center pointer-events-none z-10">
                <div className="bg-yellow-600 text-white px-6 py-2 rounded-full font-bold text-lg animate-pulse">
                  VISUALLY VERIFYING...
                </div>
              </div>
            )}
            {!analyzing && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                <p className="text-gray-500 font-mono tracking-widest uppercase">System Offline</p>
              </div>
            )}
          </div>
          <div className="h-16 bg-black/20 rounded border border-white/5 overflow-hidden">
            <canvas ref={visualizerCanvasRef} width="600" height="64" className="w-full h-full opacity-60" />
          </div>
        </div>

        <div className="glass-panel p-6 flex flex-col">
          <h2 className="text-lg font-bold mb-6 flex items-center">
            <svg className="w-5 h-5 mr-2 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            ACOUSTIC_ANALYTICS
          </h2>
          <div className="space-y-6 flex-grow">
            {Object.entries(predictions).map(([name, value]) => (
              <div key={name} className="relative">
                <div className="flex justify-between text-xs mb-1 uppercase tracking-widest font-mono">
                  <span className={name !== 'background' && value > 0.5 ? 'text-red-400' : 'text-gray-400'}>
                    {name.replace('_', ' ')}
                  </span>
                  <span className="font-bold">{(value * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${name === 'background' ? 'bg-cyan-500' : 'bg-red-500'}`}
                    style={{ width: `${value * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-8 pt-6 border-t border-white/5 text-xs text-gray-500 font-mono flex justify-between">
            <span>AI_MODEL: EMERGENCY_CNN_v2</span>
            <span>LATEST</span>
          </div>
        </div>
      </div>

      <div className="mt-12 text-center opacity-30 select-none">
        <p className="font-mono text-[10px] tracking-[0.5em] mb-2 uppercase">Antigravity Security Innovations Inc.</p>
        <div className="h-[1px] w-24 bg-cyan-500 mx-auto"></div>
      </div>
    </main>
  );
}
