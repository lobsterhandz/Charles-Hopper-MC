# utils/speech_tools.py
"""
Advanced speech_tools with Multi-Dimensional Emcee Technology:
- Monkey-patched Bark TTS safe call
- Whisper transcription
- 3D spatialization and dynamic panning
- Granular & pitch-shift effects
- Multi-voice layering
- Robust error logging and edge-case handling
"""
import os
import logging
import numpy as np
import torch
from pydub import AudioSegment, effects
from pydub.playback import play
import whisper

torch.serialization.add_safe_globals([np.core.multiarray.scalar])
# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logger = logging.getLogger("speech_tools")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Monkey-patch Bark generate_audio to drop unsupported kwargs
# ----------------------------------------------------------------------------
import bark
from bark import generate_audio as _orig_generate_audio

# Fully drop all extra args and kwargs—Bark only ever gets `text`
def _patched_generate_audio(text: str, *args, **kwargs):
    return _orig_generate_audio(text)

# Override globally
bark.generate_audio = _patched_generate_audio

# Now imports of bark.generate_audio will use the patched version
from bark import generate_audio
# In speech_tools.py, after your imports:
if torch.cuda.is_available():
    torch.set_default_device("cuda")
    logger.info("Bark TTS set to run on GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preload_models(device=device)
# ----------------------------------------------------------------------------
# Whisper transcription
# ----------------------------------------------------------------------------
# In speech_tools.py, after your imports:
if torch.cuda.is_available():
    torch.set_default_device("cuda")
    logger.info("Bark TTS set to run on GPU")
logger.info(f"Whisper device → {DEVICE}")
print("GPU available:", torch.cuda.is_available())
print("Torch device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
# Load Whisper on GPU if available
whisper_model = whisper.load_model("base", device=DEVICE)

def transcribe_user(audio_path: str) -> str:
    """Transcribe incoming user freestyle audio."""
    try:
        result = whisper_model.transcribe(audio_path)
        text = result.get("text", "")
        logger.info(f"Transcribed user audio: {text[:30]}...")
        return text
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""

# ----------------------------------------------------------------------------
# Core TTS save function
# ----------------------------------------------------------------------------
def save_bark_tts(text: str, output_path: str, **kwargs) -> str:
    """
    Synthesize text -> waveform, export to WAV.
    """
    try:
        arr = generate_audio(text, **kwargs)
        seg = AudioSegment(
            arr.tobytes(), frame_rate=24000, sample_width=2, channels=1
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        seg.export(output_path, format="wav")
        logger.info(f"Exported TTS: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"save_bark_tts failed: {e}")
        raise

# ----------------------------------------------------------------------------
# Multi-Dimensional Emcee Technology
# ----------------------------------------------------------------------------

def apply_spatial_panning(seg: AudioSegment, pan_pos: float) -> AudioSegment:
    """
    Simulate 3D panning: pan_pos ∈ [-1.0 (left) ... 1.0 (right)]
    """
    try:
        stereo = seg.set_channels(2)
        left, right = stereo.split_to_mono()
        # Linear panning
        left_gain = 1 - ((pan_pos + 1) / 2)
        right_gain = (pan_pos + 1) / 2
        left = left.apply_gain(-3 * (1 - left_gain))
        right = right.apply_gain(-3 * (1 - right_gain))
        return AudioSegment.from_mono_audiosegments(left, right)
    except Exception as e:
        logger.warning(f"Spatial panning failed: {e}")
        return seg


def granular_effect(seg: AudioSegment, grain_ms: int = 50, overlap: float = 0.5) -> AudioSegment:
    """
    Apply granular synthesis: split into grains and reassemble with overlap.
    """
    try:
        grains = []
        step = int(grain_ms * (1 - overlap))
        for i in range(0, len(seg), step):
            grain = seg[i:i+grain_ms]
            # random small pitch shift
            grain = effects.speedup(grain, playback_speed=random.uniform(0.9, 1.1))
            grains.append(grain)
        return sum(grains)
    except Exception as e:
        logger.warning(f"Granular effect failed: {e}")
        return seg


def dimension_shift(seg: AudioSegment, freq_mod: float = 0.2) -> AudioSegment:
    """
    Simulate dimension-shift by modulating pitch over time.
    """
    try:
        # create a fast vibrato effect
        modulated = seg._spawn(seg.raw_data, overrides={
            "frame_rate": int(seg.frame_rate * (1 + freq_mod))
        }).set_frame_rate(seg.frame_rate)
        return modulated
    except Exception as e:
        logger.warning(f"Dimension shift failed: {e}")
        return seg

# ----------------------------------------------------------------------------
# High-Level Multi-Voice Synthesis
# ----------------------------------------------------------------------------

def save_multidimensional_tts(
    text: str,
    output_path: str,
    layers: int = 3,
    pan_spread: float = 0.8,
    granular: bool = True,
    dimension: bool = True
) -> str:
    """
    Synthesize multi-layered, spatial, granular TTS for a futuristic emcee.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base = AudioSegment.silent(duration=0)
    for i in range(layers):
        try:
            bar = generate_audio(text)
            seg = AudioSegment(bar.tobytes(), frame_rate=24000, sample_width=2, channels=1)
            # apply spatial distribution
            pan = -pan_spread + 2*pan_spread*(i/(layers-1))
            seg = apply_spatial_panning(seg, pan)
            if granular:
                seg = granular_effect(seg)
            if dimension:
                seg = dimension_shift(seg)
            base = base.overlay(seg)
            logger.debug(f"Layer {i} applied at pan {pan}")
        except Exception as e:
            logger.error(f"Layer {i} synthesis failed: {e}")
    base = effects.normalize(base)
    base.export(output_path, format="wav")
    logger.info(f"Exported multidimensional TTS: {output_path}")
    return output_path

# ----------------------------------------------------------------------------
# Timing split remains unchanged
# ----------------------------------------------------------------------------
def split_bars_for_timing(text: str, bpm: int = 90):
    bar_ms = int((60000 / bpm) * 4)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bars = lines[:16] if len(lines)>=16 else lines + ["..."]*(16-len(lines))
    return [(bars[i], i*bar_ms) for i in range(16)]
