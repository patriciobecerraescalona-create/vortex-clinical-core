import sounddevice as sd
import numpy as np
import requests
import sys
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

STT_BASE = "http://localhost:9000"
SESSION_ID = None  # pega aquí el session_id activo
if not SESSION_ID:
    print("❌ Sesión no activa. No se graba.")
    sys.exit(1)

SAMPLE_RATE = 16000
DURATION = 5  # segundos
WAV_PATH = "input.wav"

print("🎙️ Grabando... habla ahora")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
print("✅ Grabación terminada")

write(WAV_PATH, SAMPLE_RATE, audio)

print("🧠 Transcribiendo...")
model = WhisperModel("base", compute_type="int8")
segments, info = model.transcribe(WAV_PATH, language="es")

text = " ".join([seg.text for seg in segments])
print("📝 Texto:", text)
