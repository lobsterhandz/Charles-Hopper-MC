import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

# Audio processing
import numpy as np
import librosa
from pydub import AudioSegment, effects

# TTS synthesis (from your patched speech_tools)
from utils.speech_tools import save_bark_tts  # returns path to WAV
#from utils.speech_tools import save_multidimensional_tts as save_bark_tts # if u want 3d layered version
# ----------------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------------
logger = logging.getLogger("charlie_mix")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# File handler
fh = logging.FileHandler(".codex/charlie_mix.log")
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# ----------------------------------------------------------------------------
# Utility: BPM Detection
# ----------------------------------------------------------------------------
def detect_bpm(beat_path: str) -> float:
    """
    Attempt to detect the BPM of a beat file using librosa.
    Fallback to default if detection fails.
    """
    try:
        y, sr = librosa.load(beat_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        logger.info(f"Detected BPM: {tempo:.2f}")
        return tempo
    except Exception as e:
        logger.warning(f"BPM detection failed, using default 90 BPM: {e}")
        return 90.0

# ----------------------------------------------------------------------------
# Core: Sync Bars to Beat
# ----------------------------------------------------------------------------
def sync_bars_to_beat(
    beat_path: str,
    bars_with_timing: List[Tuple[str, int]],
    output_path: str = "generated/mix_synced.wav",
    offset_ms: int = 1000,
    parallel_synthesis: bool = True,
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

    def synthesize_one(idx: int, bar: str) -> Optional[AudioSegment]:
        try:
            clip = synthesize_bark_tts(bar)
            if voice_gain_adjustment:
                clip = clip.apply_gain(voice_gain_adjustment)
            if apply_fade:
                clip = clip.fade_in(fade_duration).fade_out(fade_duration)
            logger.debug(f"Synthesized bar {idx}")
            return clip
        except Exception as e:
            logger.error(f"Bar synthesis failed idx={idx}: {e}")
            return None

    if parallel_synthesis:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(synthesize_one, i, b): i for i, (b, _) in enumerate(bars_with_timing)}
            for fut in as_completed(futures):
                i = futures[fut]
                clips[i] = fut.result()
    else:
        for i, (bar, _) in enumerate(bars_with_timing):
            clips[i] = synthesize_one(i, bar)

    # Overlay each clip
    mixed = beat[:]
    for idx, (_, start_ms) in enumerate(bars_with_timing):
        clip = clips.get(idx)
        position = offset_ms + start_ms
        if clip is None:
            logger.warning(f"Skipping overlay for bar {idx}, no clip available")
            continue
        # Trim overflow
        if position + len(clip) > len(beat):
            clip = clip[: len(beat) - position]
            logger.debug(f"Trimmed bar {idx} to fit beat length")
        try:
            mixed = mixed.overlay(clip, position=position)
            logger.info(f"Overlayed bar {idx} at {position}ms")
        except Exception as e:
            logger.error(f"Failed to overlay bar {idx}: {e}")

    # Normalize final mix
    final = effects.normalize(mixed)
    try:
        final.export(output_path, format="wav")
        logger.info(f"Exported synced mix to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export mix: {e}")
        raise

    return output_path

# ----------------------------------------------------------------------------
# Core: Mix Beat and Full Voice Clip
# ----------------------------------------------------------------------------
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

    # Loop or trim voice to beat length
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

    parser = argparse.ArgumentParser(description="Advanced Charlie Mixer")
    sub = parser.add_subparsers(dest="cmd")

    # sync command
    p1 = sub.add_parser("sync", help="Sync bars to beat")
    p1.add_argument("beat", help="Path to beat WAV file")
    p1.add_argument("bars_file", help="Text file with bar|time_ms per line")
    p1.add_argument("output", help="Output WAV path")

    # mix command
    p2 = sub.add_parser("mix", help="Mix full voice to beat")
    p2.add_argument("beat", help="Path to beat WAV file")
    p2.add_argument("voice", help="Path to voice WAV file")
    p2.add_argument("output", help="Output WAV path")

    args = parser.parse_args()

    if args.cmd == "sync":
        # Load bars list from file
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
