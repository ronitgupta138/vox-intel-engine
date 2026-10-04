from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timezone

class AcousticMetrics(BaseModel):
    duration_seconds: float = Field(..., description="Total audio duration in seconds")
    mean_pitch_hz: float = Field(..., description="Fundamental vocal frequency F0 in Hz")
    pitch_std_hz: float = Field(..., description="Pitch standard deviation indicating vocal variety vs monotony")
    pitch_min_hz: float
    pitch_max_hz: float
    rms_energy_db: float = Field(..., description="Root Mean Square energy volume level in dB")
    energy_dynamics_range: float = Field(..., description="Peak-to-average volume dynamic variation")
    total_pause_count: int = Field(..., description="Number of detected hesitation pauses (>300ms)")
    total_pause_duration_seconds: float
    pause_to_speech_ratio: float = Field(..., description="Fraction of time spent in silence/pauses")

class LexicalMetrics(BaseModel):
    total_words: int
    words_per_minute: float = Field(..., description="Pacing speed in WPM")
    pacing_category: str = Field(..., description="TOO_SLOW, IDEAL, TOO_FAST")
    filler_word_count: int
    filler_word_ratio: float = Field(..., description="Percentage of spoken words that are fillers")
    filler_words_detected: List[str]
    unique_words: int
    type_token_ratio: float = Field(..., description="Lexical diversity score (0.0 - 1.0)")
    sentiment_polarity: float = Field(..., description="Estimated sentiment valence (-1.0 to 1.0)")
    sentiment_label: str = Field(..., description="POSITIVE, NEUTRAL, NEGATIVE")

class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float
    status: str
    feedback: str

class ConfidenceReport(BaseModel):
    overall_confidence_score: float = Field(..., ge=0.0, le=100.0)
    delivery_grade: str = Field(..., description="A+, A, B, C, Needs Practice")
    dimension_scores: List[DimensionScore]
    actionable_recommendations: List[str]
    strengths: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TextAnalysisRequest(BaseModel):
    transcript: str = Field(..., min_length=5, description="Spoken transcript text to evaluate")
    duration_seconds: Optional[float] = Field(None, gt=0, description="Optional recorded duration in seconds")

class MultiModalAnalysisResponse(BaseModel):
    success: bool = True
    session_id: str
    acoustic_telemetry: Optional[AcousticMetrics] = None
    lexical_telemetry: Optional[LexicalMetrics] = None
    evaluation_report: ConfidenceReport
