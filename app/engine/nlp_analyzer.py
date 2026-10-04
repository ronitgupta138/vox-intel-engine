import re
from app.models.schemas import LexicalMetrics
from app.core.config import settings

COMMON_FILLERS = {
    "um", "uh", "er", "ah", "like", "literally", "actually", "basically",
    "you know", "i mean", "sort of", "kind of", "right", "so yeah"
}

POSITIVE_WORDS = {
    "great", "excellent", "proven", "solution", "efficient", "optimal",
    "scalable", "robust", "innovation", "growth", "high", "success",
    "confident", "seamless", "clear", "benefit", "advance", "strong"
}

NEGATIVE_WORDS = {
    "problem", "fail", "slow", "bottleneck", "weak", "bad", "loss",
    "risk", "difficult", "struggle", "error", "delay", "poor", "uncertain"
}

class LexicalNLPAnalyzer:
    """
    Analyzes spoken transcripts for pacing velocity (WPM), filler density,
    lexical richness (Type-Token Ratio), and sentiment polarity.
    """

    @classmethod
    def analyze_transcript(cls, text: str, duration_seconds: float = None) -> LexicalMetrics:
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [w for w in cleaned_text.split() if w]
        total_words = len(words)

        if total_words == 0:
            return LexicalMetrics(
                total_words=0,
                words_per_minute=0.0,
                pacing_category="TOO_SLOW",
                filler_word_count=0,
                filler_word_ratio=0.0,
                filler_words_detected=[],
                unique_words=0,
                type_token_ratio=0.0,
                sentiment_polarity=0.0,
                sentiment_label="NEUTRAL"
            )

        # Estimate duration if not provided (assume default 135 WPM pace)
        effective_duration = duration_seconds if (duration_seconds and duration_seconds > 0) else (total_words / 135.0) * 60.0
        wpm = (total_words / effective_duration) * 60.0

        if wpm < settings.IDEAL_WPM_MIN:
            pacing = "TOO_SLOW"
        elif wpm > settings.IDEAL_WPM_MAX:
            pacing = "TOO_FAST"
        else:
            pacing = "IDEAL"

        # Detect Fillers
        detected_fillers = []
        for filler in COMMON_FILLERS:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text.lower())
            if matches:
                detected_fillers.extend([filler] * len(matches))

        filler_count = len(detected_fillers)
        filler_ratio = float(filler_count / total_words)

        # Lexical Diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = float(unique_words / total_words)

        # Sentiment Analysis
        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        denom = max(1, pos_count + neg_count)
        polarity = (pos_count - neg_count) / denom

        if polarity > 0.15:
            sentiment_label = "POSITIVE"
        elif polarity < -0.15:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"

        return LexicalMetrics(
            total_words=total_words,
            words_per_minute=round(wpm, 1),
            pacing_category=pacing,
            filler_word_count=filler_count,
            filler_word_ratio=round(filler_ratio, 3),
            filler_words_detected=list(set(detected_fillers)),
            unique_words=unique_words,
            type_token_ratio=round(ttr, 3),
            sentiment_polarity=round(polarity, 2),
            sentiment_label=sentiment_label
        )
