# utils/speech_tools.py
"""
Robust speech tools:
- OpenAI TTS (primary) with gpt-4o-mini-tts
- Offline fallback via pyttsx3
- OpenAI transcription (gpt-4o-transcribe with fallback to whisper-1)
- Optional multi-layer voice effects
"""
import os
import logging
import random
from typing import Optional

from pydub import AudioSegment, effects

try:
    from openai import OpenAI
except Exception:  # library optional at import time
    OpenAI = None  # type: ignore

logger = logging.getLogger("speech_tools")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

_client: Optional["OpenAI"] = None


def _get_openai_client() -> Optional["OpenAI"]:
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:
        return None
    try:
        _client = OpenAI()
        return _client
    except Exception as e:
        logger.warning(f"OpenAI client unavailable: {e}")
        return None


def transcribe_user(audio_path: str) -> str:
    """
    Transcribe user audio using OpenAI hosted models to avoid local GPU/ffmpeg issues.
    Prefers gpt-4o-transcribe, falls back to whisper-1.
    """
    client = _get_openai_client()
    if client is None:
        logger.error("Transcription requires OPENAI_API_KEY or OpenAI client.")
        return ""
    model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
            )
        text = getattr(resp, "text", "") or ""
        logger.info(f"Transcribed user audio: {text[:60]}...")
        return text
    except Exception as e_primary:
        logger.warning(f"Transcription with {model} failed: {e_primary}")
        try:
            with open(audio_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            text = getattr(resp, "text", "") or ""
            logger.info(f"Transcribed with whisper-1: {text[:60]}...")
            return text
        except Exception as e:
            logger.error(f"All transcription backends failed: {e}")
            return ""


def save_bark_tts(
    text: str,
    output_path: str,
    *,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    speed: Optional[float] = None,
) -> str:
    """
    Synthesize text to speech.
    Primary: OpenAI gpt-4o-mini-tts (streaming) → WAV
    Fallback: offline pyttsx3 (quality lower, but reliable)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try OpenAI TTS first
    client = _get_openai_client()
    if client is not None:
        try:
            tts_model = model or os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
            tts_voice = voice or os.getenv("OPENAI_TTS_VOICE", "verse")
            kwargs = {"model": tts_model, "voice": tts_voice, "input": text}
            if speed is not None:
                kwargs["speed"] = float(speed)  # type: ignore
            with client.audio.speech.with_streaming_response.create(**kwargs) as response:
                response.stream_to_file(output_path)
            logger.info(f"Exported TTS (OpenAI) → {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"OpenAI TTS failed, falling back to pyttsx3: {e}")

    # Fallback: pyttsx3
    try:
        import pyttsx3  # lightweight, offline on Windows

        engine = pyttsx3.init()
        if speed is not None:
            try:
                rate = engine.getProperty("rate")
                engine.setProperty("rate", int(rate * float(speed)))
            except Exception:
                pass
        # Ensure .wav extension
        if not output_path.lower().endswith(".wav"):
            output_path = f"{os.path.splitext(output_path)[0]}.wav"
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        logger.info(f"Exported TTS (pyttsx3) → {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"TTS failed (no available backends): {e}")
        raise


# ---------- Optional creative effects for layered voices ----------
def apply_spatial_panning(seg: AudioSegment, pan_pos: float) -> AudioSegment:
    """
    pan_pos ∈ [-1.0 (left) ... 1.0 (right)]
    """
    try:
        return seg.pan(max(-1.0, min(1.0, pan_pos)))
    except Exception as e:
        logger.warning(f"Spatial panning failed: {e}")
        return seg


def granular_effect(seg: AudioSegment, grain_ms: int = 50, overlap: float = 0.5) -> AudioSegment:
    """
    Simple granular feel by slicing and minor speed jitter.
    """
    try:
        grains = []
        step = max(1, int(grain_ms * (1 - overlap)))
        for i in range(0, len(seg), step):
            grain = seg[i : i + grain_ms]
            speed = random.uniform(0.95, 1.05)
            grains.append(effects.speedup(grain, playback_speed=speed))
        return sum(grains) if grains else seg
    except Exception as e:
        logger.warning(f"Granular effect failed: {e}")
        return seg


def dimension_shift(seg: AudioSegment, cents: float = 20.0) -> AudioSegment:
    """
    Very light pitch shift by frame rate trick (approx).
    """
    try:
        factor = 2 ** (cents / 1200.0)
        mod = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * factor)})
        return mod.set_frame_rate(seg.frame_rate)
    except Exception as e:
        logger.warning(f"Dimension shift failed: {e}")
        return seg


def save_multidimensional_tts(
    text: str,
    output_path: str,
    layers: int = 3,
    pan_spread: float = 0.8,
    granular: bool = True,
    dimension: bool = True,
) -> str:
    """
    Multi-voice layering using the same backend TTS multiple times with effects.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base = AudioSegment.silent(duration=0)
    tmp_files = []
    try:
        for i in range(max(1, layers)):
            tmp = os.path.join(os.path.dirname(output_path), f"_tmp_layer_{i}.wav")
            save_bark_tts(text, tmp)
            seg = AudioSegment.from_wav(tmp)
            # spread across stereo field
            if layers > 1:
                pan = -pan_spread + 2 * pan_spread * (i / (layers - 1))
                seg = apply_spatial_panning(seg, pan)
            if granular:
                seg = granular_effect(seg)
            if dimension:
                seg = dimension_shift(seg, cents=random.uniform(-30, 30))
            base = base.overlay(seg)
            tmp_files.append(tmp)
        base = effects.normalize(base)
        base.export(output_path, format="wav")
        logger.info(f"Exported multidimensional TTS → {output_path}")
        return output_path
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass


def split_bars_for_timing(text: str, bpm: int = 90):
    bar_ms = int((60000 / bpm) * 4)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bars = lines[:16] if len(lines) >= 16 else lines + ["..."] * (16 - len(lines))
    return [(bars[i], i * bar_ms) for i in range(16)]