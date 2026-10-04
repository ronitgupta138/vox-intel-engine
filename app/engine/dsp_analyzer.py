import numpy as np
import io
import wave
from app.models.schemas import AcousticMetrics

class DigitalSignalProcessor:
    """
    Extracts acoustic digital signal processing metrics from raw audio PCM waveforms.
    Uses Autocorrelation and FFT for fundamental vocal frequency (F0) estimation.
    """

    @staticmethod
    def parse_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Extract raw PCM float32 array and sample rate from WAV byte payload."""
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()
                num_channels = wf.getnchannels()
                raw_data = wf.readframes(num_frames)

                if wf.getsampwidth() == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif wf.getsampwidth() == 4:
                    audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                if num_channels > 1:
                    audio = audio.reshape(-1, num_channels).mean(axis=1)

                return audio, sample_rate

    @staticmethod
    def estimate_pitch_f0(frame: np.ndarray, sample_rate: int, fmin=75, fmax=500) -> float:
        """
        Calculates fundamental frequency (F0) of an audio window via normalized autocorrelation.
        """
        if len(frame) == 0 or np.max(np.abs(frame)) < 1e-4:
            return 0.0

        # Autocorrelation via FFT
        n = len(frame)
        f_frame = np.fft.rfft(frame, n=2 * n)
        ac = np.fft.irfft(f_frame * np.conj(f_frame))[:n]

        # Lag bounds corresponding to human voice pitch [75 Hz, 500 Hz]
        min_lag = int(sample_rate / fmax)
        max_lag = int(sample_rate / fmin)

        if max_lag >= len(ac):
            max_lag = len(ac) - 1

        if min_lag >= max_lag:
            return 0.0

        peak_lag = min_lag + np.argmax(ac[min_lag:max_lag])
        if ac[0] > 0 and ac[peak_lag] / ac[0] > 0.25:  # Voiced threshold
            return float(sample_rate / peak_lag)
        return 0.0

    @classmethod
    def analyze_audio_waveform(cls, audio: np.ndarray, sample_rate: int) -> AcousticMetrics:
        duration = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
        if duration == 0:
            raise ValueError("Audio duration cannot be zero")

        # 1. RMS Energy dynamics
        frame_size = int(sample_rate * 0.03)  # 30ms frames
        hop_size = int(sample_rate * 0.015)   # 15ms hop
        
        num_frames = max(1, (len(audio) - frame_size) // hop_size)
        rms_energies = []
        pitches = []
        
        silence_threshold = 0.015
        pause_frames = 0
        total_pause_frames = 0
        pause_events = 0
        in_pause = False

        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio[start:end]
            
            # RMS
            rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            rms_energies.append(rms)

            # Silence / Pause detection (>300ms pause = 20 consecutive 15ms frames)
            if rms < silence_threshold:
                pause_frames += 1
                total_pause_frames += 1
                if pause_frames >= 20 and not in_pause:
                    pause_events += 1
                    in_pause = True
            else:
                pause_frames = 0
                in_pause = False

            # Pitch tracking
            f0 = cls.estimate_pitch_f0(frame, sample_rate)
            if f0 > 0:
                pitches.append(f0)

        rms_arr = np.array(rms_energies)
        mean_rms = np.mean(rms_arr)
        rms_db = float(20 * np.log10(mean_rms + 1e-6))
        energy_dynamics = float(np.max(rms_arr) - np.min(rms_arr))

        if pitches:
            mean_pitch = float(np.mean(pitches))
            pitch_std = float(np.std(pitches))
            min_pitch = float(np.min(pitches))
            max_pitch = float(np.max(pitches))
        else:
            mean_pitch = 120.0
            pitch_std = 0.0
            min_pitch = 120.0
            max_pitch = 120.0

        pause_duration = float(total_pause_frames * 0.015)
        pause_ratio = float(min(1.0, pause_duration / duration))

        return AcousticMetrics(
            duration_seconds=round(duration, 2),
            mean_pitch_hz=round(mean_pitch, 1),
            pitch_std_hz=round(pitch_std, 1),
            pitch_min_hz=round(min_pitch, 1),
            pitch_max_hz=round(max_pitch, 1),
            rms_energy_db=round(rms_db, 1),
            energy_dynamics_range=round(energy_dynamics, 3),
            total_pause_count=pause_events,
            total_pause_duration_seconds=round(pause_duration, 2),
            pause_to_speech_ratio=round(pause_ratio, 3)
        )

    @classmethod
    def generate_synthetic_test_wav(cls, duration=3.0, freq=150.0, sample_rate=16000) -> bytes:
        """Creates an in-memory 16-bit mono WAV buffer for testing and verification."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * freq * t)
        # Add intermittent modulation
        waveform *= (0.7 + 0.3 * np.sin(2 * np.pi * 1.5 * t))
        audio_int16 = (waveform * 32767).astype(np.int16)

        bio = io.BytesIO()
        with wave.open(bio, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return bio.getvalue()
