import pytest
import numpy as np
from app.engine.dsp_analyzer import DigitalSignalProcessor
from app.engine.nlp_analyzer import LexicalNLPAnalyzer
from app.engine.fusion_engine import MultiModalFusionEngine
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_dsp_synthetic_wav_analysis():
    wav_bytes = DigitalSignalProcessor.generate_synthetic_test_wav(duration=2.0, freq=160.0)
    audio, sr = DigitalSignalProcessor.parse_wav_bytes(wav_bytes)
    assert sr == 16000
    assert len(audio) == 32000

    metrics = DigitalSignalProcessor.analyze_audio_waveform(audio, sr)
    assert metrics.duration_seconds == 2.0
    assert 130.0 <= metrics.mean_pitch_hz <= 180.0
    assert metrics.rms_energy_db < 0.0

def test_nlp_transcript_analyzer():
    text = "Good morning everyone. Today I am going to present our high-performance distributed architecture. Actually, like, we achieved seamless scalability and proven reliability."
    metrics = LexicalNLPAnalyzer.analyze_transcript(text, duration_seconds=10.0)

    assert metrics.total_words > 15
    assert metrics.words_per_minute > 0
    assert metrics.filler_word_count >= 2
    assert "actually" in metrics.filler_words_detected or "like" in metrics.filler_words_detected
    assert metrics.type_token_ratio > 0.6
    assert metrics.sentiment_label in ["POSITIVE", "NEUTRAL"]

def test_multimodal_fusion_scoring():
    wav_bytes = DigitalSignalProcessor.generate_synthetic_test_wav(duration=3.0, freq=140.0)
    audio, sr = DigitalSignalProcessor.parse_wav_bytes(wav_bytes)
    acoustic = DigitalSignalProcessor.analyze_audio_waveform(audio, sr)

    text = "We designed an optimal low-latency solution with robust fault tolerance."
    lexical = LexicalNLPAnalyzer.analyze_transcript(text, duration_seconds=3.0)

    report = MultiModalFusionEngine.evaluate(acoustic=acoustic, lexical=lexical)
    assert 0.0 <= report.overall_confidence_score <= 100.0
    assert len(report.dimension_scores) == 4
    assert len(report.strengths) > 0

def test_api_endpoints():
    # Test Health
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "UP"

    # Test Text Analysis
    res = client.post("/api/v1/analyze/text", json={
        "transcript": "In this project we demonstrate distributed systems and zero-latency consensus algorithms.",
        "duration_seconds": 5.0
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["evaluation_report"]["overall_confidence_score"] > 50.0

    # Test Audio Analysis
    wav_bytes = DigitalSignalProcessor.generate_synthetic_test_wav(duration=2.0)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    res = client.post("/api/v1/analyze/audio", files=files, data={"transcript": "Testing presentation confidence analysis."})
    assert res.status_code == 200
    data = res.json()
    assert data["acoustic_telemetry"]["mean_pitch_hz"] > 0
