from app.models.schemas import (
    AcousticMetrics, LexicalMetrics, ConfidenceReport, DimensionScore
)

class MultiModalFusionEngine:
    """
    Fuses acoustic digital signal features and lexical NLP metrics into a
    calibrated multi-dimensional presentation delivery score (0-100).
    """

    @classmethod
    def evaluate(cls, acoustic: AcousticMetrics = None, lexical: LexicalMetrics = None) -> ConfidenceReport:
        dimensions = []
        recommendations = []
        strengths = []

        # 1. Pacing & Tempo Dimension (Weight: 25%)
        if lexical:
            wpm = lexical.words_per_minute
            if 125 <= wpm <= 155:
                pacing_score = 95.0
                pacing_status = "EXCELLENT"
                strengths.append(f"Ideal speaking tempo at {wpm} WPM (clear and easy to follow).")
            elif 110 <= wpm < 125 or 155 < wpm <= 175:
                pacing_score = 80.0
                pacing_status = "GOOD"
            elif wpm < 110:
                pacing_score = 55.0
                pacing_status = "TOO_SLOW"
                recommendations.append("Increase speech tempo to 130-150 WPM to maintain listener engagement.")
            else:
                pacing_score = 50.0
                pacing_status = "RUSHED"
                recommendations.append("Slow down slightly; speaking over 165 WPM can hinder technical comprehension.")

            dimensions.append(DimensionScore(
                dimension="Pacing & Cadence",
                score=pacing_score,
                weight=0.25,
                status=pacing_status,
                feedback=f"Speaking pace clocked at {wpm} WPM ({lexical.pacing_category})."
            ))

        # 2. Fluency & Articulation Dimension (Weight: 25%)
        if lexical:
            filler_ratio = lexical.filler_word_ratio
            if filler_ratio <= 0.015:
                fluency_score = 96.0
                fluency_status = "SUPERB"
                strengths.append("Exceptional vocal fluency with minimal filler word crutches.")
            elif filler_ratio <= 0.035:
                fluency_score = 82.0
                fluency_status = "COMPETENT"
            else:
                fluency_score = max(35.0, 100.0 - (filler_ratio * 1200))
                fluency_status = "HIGH_FILLERS"
                recommendations.append(f"Reduce filler words ({', '.join(lexical.filler_words_detected[:4])}). Replace with deliberate micro-pauses.")

            dimensions.append(DimensionScore(
                dimension="Fluency & Articulation",
                score=round(fluency_score, 1),
                weight=0.25,
                status=fluency_status,
                feedback=f"Filler word ratio measured at {round(filler_ratio * 100, 1)}% ({lexical.filler_word_count} occurrences)."
            ))

        # 3. Vocal Energy & Projection (Weight: 25%)
        if acoustic:
            db = acoustic.rms_energy_db
            dynamics = acoustic.energy_dynamics_range

            if -28 <= db <= -12 and dynamics > 0.08:
                energy_score = 92.0
                energy_status = "STRONG"
                strengths.append("Resonant voice projection with engaging dynamic volume emphasis.")
            elif -35 <= db < -28:
                energy_score = 75.0
                energy_status = "MODERATE"
            else:
                energy_score = 55.0
                energy_status = "LOW_PROJECTION"
                recommendations.append("Speak closer to the microphone and project from the diaphragm for greater vocal presence.")

            dimensions.append(DimensionScore(
                dimension="Vocal Energy & Projection",
                score=energy_score,
                weight=0.25,
                status=energy_status,
                feedback=f"Volume level at {db} dB with dynamic range of {dynamics}."
            ))

        # 4. Pitch Modulation & Intonation Variety (Weight: 25%)
        if acoustic:
            pitch_std = acoustic.pitch_std_hz
            if 18.0 <= pitch_std <= 45.0:
                pitch_score = 94.0
                pitch_status = "EXPRESSIVE"
                strengths.append("Rich pitch modulation; lively intonation avoids sounding monotone.")
            elif pitch_std < 18.0:
                pitch_score = 60.0
                pitch_status = "MONOTONE"
                recommendations.append("Vary vocal pitch across key statements to emphasize critical technical concepts.")
            else:
                pitch_score = 75.0
                pitch_status = "HIGH_VARIANCE"

            dimensions.append(DimensionScore(
                dimension="Intonation & Expression",
                score=pitch_score,
                weight=0.25,
                status=pitch_status,
                feedback=f"Fundamental pitch standard deviation: {pitch_std} Hz around mean {acoustic.mean_pitch_hz} Hz."
            ))

        # Compute Weighted Score
        if dimensions:
            total_weight = sum(d.weight for d in dimensions)
            overall_score = sum(d.score * d.weight for d in dimensions) / total_weight
        else:
            overall_score = 75.0

        overall_score = round(overall_score, 1)

        # Assign Grade
        if overall_score >= 90.0:
            grade = "A+ (Executive Ready)"
        elif overall_score >= 80.0:
            grade = "A (Strong Delivery)"
        elif overall_score >= 70.0:
            grade = "B (Competent)"
        elif overall_score >= 60.0:
            grade = "C (Needs Polish)"
        else:
            grade = "Needs Practice"

        if not recommendations:
            recommendations.append("Maintain consistent pace and posture throughout high-stakes Q&A rounds.")

        return ConfidenceReport(
            overall_confidence_score=overall_score,
            delivery_grade=grade,
            dimension_scores=dimensions,
            actionable_recommendations=recommendations,
            strengths=strengths
        )
