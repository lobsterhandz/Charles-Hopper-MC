# utils/audio_tools.py
"""
Overlay utilities for mixing TTS bars and full voice onto a beat, with advanced features:
- BPM detection via librosa
- Concurrent TTS synthesis for speed
- Fade, gain adjustment, overflow trimming
- Robust logging and CLI interface
"""
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
# Optional dependency: librosa (lazy-guarded)
try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore
from pydub import AudioSegment, effects

from utils.speech_tools import save_bark_tts, apply_spatial_panning, granular_effect, dimension_shift  # writes out TTS WAV files

# ZeroGPU integration (no-op outside Spaces)
try:
    import spaces  # type: ignore
    GPU_DECORATOR = spaces.GPU
except Exception:
    def GPU_DECORATOR(*args, **kwargs):
        def _no_gpu(fn):
            return fn
        return _no_gpu

# ----------------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------------
logger = logging.getLogger("audio_tools")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ----------------------------------------------------------------------------
# Utility: BPM Detection
# ----------------------------------------------------------------------------
def detect_bpm(beat_path: str) -> float:
    """
    Attempt to detect the BPM of a beat file using librosa.
    Fallback to default if detection fails or librosa is unavailable.
    """
    try:
        if librosa is None:
            logger.warning("librosa not installed; using default 90 BPM")
            return 90.0
        y, sr = librosa.load(beat_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        logger.info(f"Detected BPM: {tempo:.2f}")
        return float(tempo)
    except Exception as e:
        logger.warning(f"BPM detection failed, using default 90 BPM: {e}")
        return 90.0

# ----------------------------------------------------------------------------
# Emotion processing helpers
# ----------------------------------------------------------------------------
def parse_emotion_and_text(bar_text: str):
    """
    Extract leading emotion tag like [angry] and return (emotion, clean_text).
    Falls back to 'neutral' when absent.
    """
    text = bar_text.strip()
    emo = "neutral"
    if text.startswith("[") and "]" in text:
        try:
            tag = text.split("]", 1)[0].strip("[]").lower()
            if tag in ("angry", "brag", "reflective", "neutral"):
                emo = tag
                text = text.split("]", 1)[1].strip()
        except Exception:
            pass
    return emo, text

def add_room_reverb(seg: AudioSegment, taps: list[tuple[int, float]]):
    """
    Very light room reverb via multi-tap delays with gentle low-pass filtering.
    taps: list of (delay_ms, gain_db)
    """
    out = seg
    for delay_ms, gain_db in taps:
        tap = seg.apply_gain(gain_db)
        tap = effects.low_pass_filter(tap, 4500)
        out = out.overlay(tap, position=int(delay_ms))
    return out

EMOTION_PARAMS = {
    "angry":      {"speed": 1.08, "cents":  25.0, "pan": -0.05, "duck_db": 7.0, "reverb": [(40, -14.0), (85, -18.0)]},
    "brag":       {"speed": 0.98, "cents": -10.0, "pan":  0.15, "duck_db": 5.0, "reverb": [(60, -16.0), (120, -20.0)]},
    "reflective": {"speed": 0.95, "cents": -15.0, "pan": -0.10, "duck_db": 4.0, "reverb": [(90, -14.0), (180, -18.0)]},
    "neutral":    {"speed": 1.00, "cents":   0.0, "pan":  0.00, "duck_db": 5.0, "reverb": [(70, -18.0)]},
}

# ----------------------------------------------------------------------------
# Core: Sync Bars to Beat
# ----------------------------------------------------------------------------
@GPU_DECORATOR(duration=120)
def sync_bars_to_beat(
    beat_path: str,
    bars_with_timing: List[Tuple[str, int]],
    output_path: str = "generated/mix_synced.wav",
    offset_ms: int = 1000,
    parallel_synthesis: bool = False,
    max_workers: int = 4,
    apply_fade: bool = True,
    fade_duration: int = 50,
    voice_gain_adjustment: float = 0.0
) -> str:
    """
    Overlay synthesized bars onto the beat with timing, optional concurrency,
    fade, gain, and robust error handling.

    Returns the output_path on success.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load base beat
    try:
        beat = AudioSegment.from_file(beat_path)
        logger.info(f"Loaded beat: {beat_path}, duration {len(beat)}ms")
    except Exception as e:
        logger.error(f"Failed to load beat '{beat_path}': {e}")
        raise

    # Pre-synthesize bars
    logger.info("Starting bar synthesis...")
    clips = {}

    def synthesize_one(idx: int, bar_text: str):
        try:
            emo, clean_text = parse_emotion_and_text(bar_text)
            params = EMOTION_PARAMS.get(emo, EMOTION_PARAMS["neutral"])

            wav_path = f"generated/bar_{idx}.wav"
            # Emotion-driven delivery speed
            save_bark_tts(clean_text, wav_path, speed=float(params.get("speed", 1.0)))
            clip = AudioSegment.from_wav(wav_path)

            # Spatial and timbre tweaks
            clip = apply_spatial_panning(clip, float(params.get("pan", 0.0)))
            cents = float(params.get("cents", 0.0))
            if abs(cents) > 1e-3:
                clip = dimension_shift(clip, cents=cents)

            # Touch of room to feel live
            reverb_taps = params.get("reverb", [])
            if reverb_taps:
                clip = add_room_reverb(clip, reverb_taps)

            if voice_gain_adjustment:
                clip = clip.apply_gain(voice_gain_adjustment)
            if apply_fade:
                clip = clip.fade_in(fade_duration).fade_out(fade_duration)

            logger.debug(f"Synthesized bar {idx} ({emo})")
            # Return clip and per-bar ducking amount
            return clip, float(params.get("duck_db", 5.0))
        except Exception as e:
            logger.error(f"Bar synthesis failed idx={idx}: {e}")
            return None

    # Synthesize in parallel or sequentially
    if parallel_synthesis:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(synthesize_one, i, b): i for i, (b, _) in enumerate(bars_with_timing)}
            for future in as_completed(future_map):
                idx = future_map[future]
                clips[idx] = future.result()
    else:
        for i, (bar_text, _) in enumerate(bars_with_timing):
            clips[i] = synthesize_one(i, bar_text)

    # Overlay each clip with gentle ducking under vocals
    mixed = beat[:]
    for idx, (bar_text, start_ms) in enumerate(bars_with_timing):
        pair = clips.get(idx)
        position = offset_ms + start_ms
        if pair is None:
            logger.warning(f"Skipping overlay for bar {idx}, no clip available")
            continue
        # Unpack synthesized clip and per-bar ducking amount
        clip, duck_db = pair

        # Trim overflow
        if position + len(clip) > len(beat):
            clip = clip[: len(beat) - position]
            logger.debug(f"Trimmed bar {idx} to fit beat length")

        # Apply sidechain-like ducking to the beat under the vocal
        region_end = min(position + len(clip), len(mixed))
        if region_end > position:
            try:
                under = mixed[position:region_end].apply_gain(-float(duck_db))
                mixed = mixed[:position] + under + mixed[region_end:]
            except Exception as e:
                logger.debug(f"Ducking failed for bar {idx}: {e}")

        try:
            mixed = mixed.overlay(clip, position=position)
            logger.info(f"Overlayed bar {idx} at {position}ms")
        except Exception as e:
            logger.error(f"Failed to overlay bar {idx}: {e}")

    # Normalize and export
    final_mix = effects.normalize(mixed)
    try:
        final_mix.export(output_path, format="wav")
        logger.info(f"Exported synced mix to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export mix: {e}")
        raise

    return output_path

# ----------------------------------------------------------------------------
# Core: Mix Beat and Full Voice Clip
# ----------------------------------------------------------------------------
@GPU_DECORATOR(duration=90)
def mix_beat_and_voice(
    beat_path: str,
    voice_path: str,
    output_path: str = "generated/mix.wav",
    offset_ms: int = 500,
    normalize_mix: bool = True,
    voice_gain_adjustment: float = 0.0
) -> str:
    """
    Overlay a full voice performance onto the beat with offset and normalization.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        beat = AudioSegment.from_file(beat_path)
        voice = AudioSegment.from_file(voice_path)
    except Exception as e:
        logger.error(f"Loading audio files failed: {e}")
        raise

    if voice_gain_adjustment:
        voice = voice.apply_gain(voice_gain_adjustment)

    # Loop or trim voice
    if len(voice) < len(beat):
        looped = AudioSegment.silent(duration=offset_ms) + voice
        while len(looped) < len(beat):
            looped += voice
        voice = looped[: len(beat)]
        logger.debug("Looped voice to match beat length")
    else:
        voice = voice[: len(beat)]
        logger.debug("Trimmed voice to beat length")

    try:
        mix = beat.overlay(voice, position=offset_ms)
        logger.info(f"Overlayed full voice at {offset_ms}ms")
    except Exception as e:
        logger.error(f"Overlay failed: {e}")
        raise

    if normalize_mix:
        mix = effects.normalize(mix)
        logger.debug("Normalized final mix")

    try:
        mix.export(output_path, format="wav")
        logger.info(f"Exported final mix to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export final mix: {e}")
        raise

    return output_path

# ----------------------------------------------------------------------------
# Optional: CLI Interface for Testing
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Audio Tools for Charlie MC")
    sub = parser.add_subparsers(dest="cmd")

    p_sync = sub.add_parser("sync", help="Sync bars to beat")
    p_sync.add_argument("beat", help="Path to beat WAV file")
    p_sync.add_argument("bars_file", help="Text file with bar|time_ms per line")
    p_sync.add_argument("output", help="Output WAV path")

    p_mix = sub.add_parser("mix", help="Mix full voice to beat")
    p_mix.add_argument("beat", help="Path to beat WAV file")
    p_mix.add_argument("voice", help="Path to voice WAV file")
    p_mix.add_argument("output", help="Output WAV path")

    args = parser.parse_args()
    if args.cmd == "sync":
        bars = []
        with open(args.bars_file) as f:
            for line in f:
                if "|" in line:
                    text, t = line.strip().split("|", 1)
                    bars.append((text, int(t)))
        sync_bars_to_beat(args.beat, bars, output_path=args.output)
    elif args.cmd == "mix":
        mix_beat_and_voice(args.beat, args.voice, output_path=args.output)
    else:
        parser.print_help()
