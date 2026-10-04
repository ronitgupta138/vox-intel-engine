<div align="center">

# 🎙️ VoxIntel Engine

**Real-Time Multi-Modal Speech Telemetry & Acoustic DSP Engine (FastAPI, Python 3.11, Autocorrelation Pitch, FFT Spectral Energy & Lexical Analytics)**

[![Python](https://img.shields.io/badge/Python-3.11%2B-0891b2?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-0891b2?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NumPy](https://img.shields.io/badge/NumPy-DSP-0891b2?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-0891b2?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-0891b2?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/ronitgupta138/vox-intel-engine/actions)

</div>

---

## 📌 Architectural Overview

VoxIntel Engine is an asynchronous, high-throughput microservice that analyzes technical presentations, executive pitches, and mock interview speeches across multiple parallel acoustic and semantic streams:

```
                           [ Audio Stream (WAV/PCM) + Transcript ]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     ┌───────────────────────────────┐                 ┌───────────────────────────────┐
     │   Acoustic DSP Analyzer       │                 │   Lexical NLP Analyzer        │
     │ - FFT Fundamental Pitch (F0)  │                 │ - Words Per Minute (WPM)      │
     │ - RMS Energy & Volume (dB)    │                 │ - Filler Word Density         │
     │ - Silence / Hesitation Pauses │                 │ - Type-Token Ratio (TTR)      │
     └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │   Multi-Modal Fusion Engine     │
                             │ - Calibrated Scoring Matrix     │
                             │ - Delivery Grade (A+ to C)      │
                             │ - Actionable Coaching Insights  │
                             └─────────────────────────────────┘
```

---

## ⚡ Core Telemetry & Mathematical Capabilities

1. **Acoustic Digital Signal Processing:**
   * **Fundamental Frequency ($F_0$):** Normalized autocorrelation over Fourier transform frames estimating vocal pitch register ($75	ext{ Hz} - 500	ext{ Hz}$).
   * **Intonation Variety & Pitch Jitter ($\sigma_{F_0}$):** Measures standard deviation around mean pitch to detect flat/monotone delivery vs expressive presentation style.
   * **Root Mean Square (RMS) Volume Dynamics:** Decibel-scale loudness measurement ($20 \log_{10}(	ext{RMS})$) and peak-to-floor dynamic contrast.
   * **Micro-Pause Segmentation:** Automated detection of hesitation gaps ($>300	ext{ms}$) vs natural rhythmic pauses.

2. **Lexical & NLP Metrics:**
   * **Speaking Pace (WPM):** Classifies cadence against standard presentation benchmarks ($125 - 155	ext{ WPM}$ ideal).
   * **Filler Word Density:** Regex token analysis detecting common hesitation crutches (`um`, `uh`, `like`, `actually`, `basically`, `you know`).
   * **Lexical Diversity (Type-Token Ratio):** Vocabulary richness calculation ($TTR = rac{	ext{Unique Tokens}}{	ext{Total Tokens}}$).

3. **Multi-Modal Confidence Matrix:**
   * Weighted multi-dimensional scoring evaluating **Pacing (25%)**, **Fluency (25%)**, **Vocal Energy (25%)**, and **Intonation (25%)** to output an overall calibrated confidence index ($0 - 100$).

---

## 📡 REST API Reference

### 1. Multi-Modal Audio Analysis
`POST /api/v1/analyze/audio` (Multipart Form)
* **`file`**: Audio recording (`.wav`)
* **`transcript`** *(optional)*: Spoken text transcript

### 2. Transcript-Only Pacing Analysis
`POST /api/v1/analyze/text` (JSON Body)
```json
{
  "transcript": "Good morning. In this presentation, I will discuss distributed consensus algorithms and our high-throughput transaction architecture.",
  "duration_seconds": 15.0
}
```

### 3. Health & Telemetry
`GET /api/v1/health`

---

## 🚀 Quick Start (Docker)

```bash
git clone https://github.com/ronitgupta138/vox-intel-engine.git
cd vox-intel-engine

# Start service container
docker compose up --build -d

# Interactive Swagger UI available at:
# http://localhost:8000/docs
```

---

## 🧪 Running Unit Tests

```bash
pytest -v tests/
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
