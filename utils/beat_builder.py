import os
import logging
import random
from typing import Optional
from pydub import AudioSegment
from pydub.generators import Sine

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------
logger = logging.getLogger("beat_builder")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# File handler for deep logs
os.makedirs('.codex', exist_ok=True)
fh = logging.FileHandler(".codex/beat_builder.log")
fh.setLevel(logging.DEBUG)
fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# --------------------------------------------------
# Beat Builder
# --------------------------------------------------
def build_beat(
    output_path: str = "generated/beat.wav",
    sample_dir: str = "assets/samples/",
    drums_dir: str = "assets/drums/",
    bpm: int = 90,
    bars: int = 16,
    swing: int = 20,
    include_melody: bool = True
) -> str:
    """
    Generate a procedural 4/4 beat with optional melodic overlay.

    Args:
        output_path: Path for exported beat WAV.
        sample_dir: Dir for melodic samples (.wav).
        drums_dir: Dir for drum samples (.wav).
        bpm: Tempo in beats-per-minute.
        bars: Number of 4/4 bars.
        swing: Maximum random offset (ms) for swing feel.
        include_melody: Whether to layer melodic samples.

    Returns:
        The resolved output_path.
    """
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, sample_dir)
    drums_dir = os.path.join(base_dir, drums_dir)
    output_path = os.path.join(base_dir, output_path)

    logger.info(f"Building beat: {bpm} BPM, {bars} bars, swing ±{swing}ms")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Helper: load sample or fallback
    def _load(dir_path: str, keyword: str) -> Optional[AudioSegment]:
        if not os.path.isdir(dir_path):
            return None
        candidates = [f for f in os.listdir(dir_path)
                      if f.lower().endswith('.wav') and keyword in f.lower()]
        if not candidates:
            return None
        choice = random.choice(candidates)
        path = os.path.join(dir_path, choice)
        try:
            seg = AudioSegment.from_wav(path)
            logger.debug(f"Loaded {keyword} sample: {choice}")
            return seg
        except Exception as e:
            logger.warning(f"Failed to load sample {path}: {e}")
            return None

    # Fallback synthesized tones
    fallback_kick = Sine(60).to_audio_segment(duration=150).apply_gain(-2)
    fallback_snare = Sine(200).to_audio_segment(duration=150).apply_gain(-6)
    fallback_hat = Sine(8000).to_audio_segment(duration=80).apply_gain(-12)

    # Load or fallback drums
    kick = _load(drums_dir, 'kick') or fallback_kick
    snare = _load(drums_dir, 'snare') or fallback_snare
    hat   = _load(drums_dir, 'hat') or fallback_hat

    # Optional melodic sample
    melody = AudioSegment.silent(duration=0)
    if include_melody:
        mel = _load(sample_dir, '')  # any .wav
        if mel:
            melody = mel.apply_gain(-8)
            logger.info("Melodic sample loaded and applied")

    # Calculate durations
    bar_ms = int((60000 / bpm) * 4)
    total_ms = bar_ms * bars
    beat = AudioSegment.silent(duration=total_ms)

    # Build bars
    for bar_idx in range(bars):
        base = bar_idx * bar_ms
        for step in range(4):
            pos = base + int(step * (bar_ms / 4)) + random.randint(-swing, swing)
            # hi-hat on every beat
            beat = beat.overlay(hat, position=max(0, pos))
            # kick on 1 and optional on 3
            if step == 0 or (step == 2 and random.random() < 0.6):
                beat = beat.overlay(kick, position=max(0, pos + random.randint(-swing//2, swing//2)))
            # snare on 2 and 4, ghost on 3
            if step in [1, 3]:
                beat = beat.overlay(snare, position=max(0, pos + random.randint(-swing//2, swing//2)))
            elif step == 2 and random.random() < 0.3:
                beat = beat.overlay(snare.apply_gain(-5), position=max(0, pos))

    # Overlay melody in loop if present
    if len(melody) > 0:
        loops = total_ms // len(melody) + 1
        full_melody = (melody * loops)[:total_ms].fade_in(200).fade_out(200)
        beat = beat.overlay(full_melody)

    try:
        beat.export(output_path, format='wav')
        logger.info(f"Beat exported: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export beat: {e}")
        raise

    return output_path
