import os
import logging
import random
from typing import Optional
from pydub import AudioSegment, effects
from pydub.generators import Sine, WhiteNoise

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
    # Resolve absolute paths (project root, not utils/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_dir = os.path.join(project_root, sample_dir)
    drums_dir = os.path.join(project_root, drums_dir)
    output_path = os.path.join(project_root, output_path)

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

    # Fallback synthesized drums (modeled for realism)
    def _synth_kick() -> AudioSegment:
        body = Sine(60).to_audio_segment(duration=140).apply_gain(-1).fade_out(120)
        sub = Sine(45).to_audio_segment(duration=160).apply_gain(-4).fade_out(140)
        click = Sine(2000).to_audio_segment(duration=6).apply_gain(-10)
        kick = body.overlay(sub).overlay(click)
        return effects.low_pass_filter(kick, 250)

    def _synth_snare() -> AudioSegment:
        noise = WhiteNoise().to_audio_segment(duration=120).apply_gain(-8)
        noise = effects.high_pass_filter(noise, 3000)
        body = Sine(190).to_audio_segment(duration=120).apply_gain(-10).fade_out(100)
        return noise.overlay(body)

    def _synth_hat() -> AudioSegment:
        noise = WhiteNoise().to_audio_segment(duration=70).apply_gain(-18)
        return effects.high_pass_filter(noise, 6000).fade_out(50)

    fallback_kick = _synth_kick()
    fallback_snare = _synth_snare()
    fallback_hat = _synth_hat()

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

    # Add a simple bassline for realism (minor pentatonic around ~A/B/C/D)
    try:
        root_candidates = [55.0, 62.0, 65.4, 73.4]  # Hz
        f0 = random.choice(root_candidates)
        minor_pent = [1.0, 1.189, 1.335, 1.5, 1.782]  # 1, b3, 4, 5, b7
        for bar_idx in range(bars):
            base = bar_idx * bar_ms
            # Two notes per bar (downbeats)
            for step in (0, 2):
                degree = (0 if step == 0 else 3)  # root then fifth
                if random.random() < 0.25:
                    degree = random.randrange(0, len(minor_pent))
                freq = f0 * minor_pent[degree]
                note_ms = int(bar_ms / 2.2)
                note = Sine(freq).to_audio_segment(duration=note_ms).apply_gain(-14).fade_in(10).fade_out(60)
                note = effects.low_pass_filter(note, 200)
                pos = base + int(step * (bar_ms / 4)) + random.randint(-swing//2, swing//2)
                beat = beat.overlay(note, position=max(0, pos))
        # Light glue: tiny master fade
        beat = beat.fade_in(50).fade_out(100)
    except Exception as e:
        logger.warning(f"Bassline synthesis failed: {e}")

    # Add simple drummer's fills on every 4 bars (hat rolls + snare flam)
    try:
        for bar_idx in range(bars):
            if (bar_idx + 1) % 4 != 0:
                continue
            bar_start = bar_idx * bar_ms
            # Hat roll on the last quarter
            roll_start = bar_start + int(0.75 * bar_ms)
            step = int(bar_ms / 24)  # 1/24th notes
            for i in range(6):
                beat = beat.overlay(hat.apply_gain(-6), position=roll_start + i * step)
            # Snare flam near the end
            flam_pos = bar_start + int(0.96 * bar_ms)
            beat = beat.overlay(snare.apply_gain(-3), position=flam_pos)
            beat = beat.overlay(snare.apply_gain(-8), position=flam_pos + 12)
    except Exception as e:
        logger.warning(f"Drum fill synthesis failed: {e}")

    try:
        beat.export(output_path, format='wav')
        logger.info(f"Beat exported: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export beat: {e}")
        raise

    return output_path
