from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import uuid
from typing import Optional

from app.engine.dsp_analyzer import DigitalSignalProcessor
from app.engine.nlp_analyzer import LexicalNLPAnalyzer
from app.engine.fusion_engine import MultiModalFusionEngine
from app.models.schemas import (
    TextAnalysisRequest, MultiModalAnalysisResponse, ConfidenceReport
)

router = APIRouter()

@router.post("/analyze/text", response_model=MultiModalAnalysisResponse)
async def analyze_transcript_endpoint(payload: TextAnalysisRequest):
    """
    Evaluates speech pacing (WPM), filler words, and lexical delivery from a transcript.
    """
    session_id = str(uuid.uuid4())[:8]
    lexical = LexicalNLPAnalyzer.analyze_transcript(payload.transcript, payload.duration_seconds)
    report = MultiModalFusionEngine.evaluate(acoustic=None, lexical=lexical)

    return MultiModalAnalysisResponse(
        success=True,
        session_id=session_id,
        acoustic_telemetry=None,
        lexical_telemetry=lexical,
        evaluation_report=report
    )

@router.post("/analyze/audio", response_model=MultiModalAnalysisResponse)
async def analyze_audio_endpoint(
    file: UploadFile = File(..., description="WAV Audio Recording of Presentation Speech"),
    transcript: Optional[str] = Form(None, description="Optional accompanying transcript text")
):
    """
    Analyzes an audio file for pitch dynamics, vocal energy (dB), pauses, and fusion confidence.
    """
    session_id = str(uuid.uuid4())[:8]
    content = await file.read()

    try:
        audio, sr = DigitalSignalProcessor.parse_wav_bytes(content)
        acoustic = DigitalSignalProcessor.analyze_audio_waveform(audio, sr)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process WAV audio: {str(e)}"
        )

    lexical = None
    if transcript and transcript.strip():
        lexical = LexicalNLPAnalyzer.analyze_transcript(transcript, acoustic.duration_seconds)

    report = MultiModalFusionEngine.evaluate(acoustic=acoustic, lexical=lexical)

    return MultiModalAnalysisResponse(
        success=True,
        session_id=session_id,
        acoustic_telemetry=acoustic,
        lexical_telemetry=lexical,
        evaluation_report=report
    )

@router.get("/health")
async def health_check():
    return {
        "status": "UP",
        "service": "Presentation-Intelligence-Engine",
        "pipeline": "DSP-Audio + NLP-Lexical + Weighted-Fusion-Matrix",
        "version": "1.0.0"
    }
