from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI(title="STT Service")

model = WhisperModel("base", compute_type="int8")

class AudioText(BaseModel):
    text: str


@app.post("/stt/transcribe", response_model=AudioText)
def transcribe(text: AudioText):
    """
    LAB: simula transcripción.
    En prod aquí entra el audio real.
    """
    return text
