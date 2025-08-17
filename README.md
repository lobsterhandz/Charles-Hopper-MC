---
title: "Charles Hopper MC"
emoji: "🎤"
colorFrom: "gray"
colorTo: "yellow"
sdk: gradio
python_version: 3.10.13
suggested_hardware: zero-a10g
sdk_version: 5.25.0
app_file: app.py
license: mit
tags:
  - gradio
  - audio
  - music
  - hip-hop
  - beatmaker
  - speech-to-text
  - tts
  - ai
  - demo
---

# Charles Hopper MC

[![Gradio App](https://img.shields.io/badge/Gradio-App-orange?logo=Gradio&logoColor=white&style=for-the-badge)](https://gradio.app/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

Live battle rap experience: assemble a beat, freestyle into the mic, and get a ruthless AI rebuttal. Every run is varied for a fun, engaging session.

## What’s New

Stability and variety improvements for Windows and Spaces:

- Hosted OpenAI TTS/STT (with offline TTS fallback): reliable, no GPU or local Whisper required
- Robust audio save/format handling for Gradio v5/v4 microphone input
- Procedural beat builder with project-root asset resolution
- Randomized BPM and sync offset each round; randomized style directives per verse for fresh bars
- Minimal, clean requirements and optional librosa
- Optional multi-voice layering/effects utilities for wild performances

## Quickstart

1) Install Python 3.10–3.12

2) Install ffmpeg (required by pydub)
- Windows (PowerShell):
  - winget:
    winget install --id=Gyan.FFmpeg -e --source winget
  - or Chocolatey:
    choco install ffmpeg
  - Verify:
    ffmpeg -version
- macOS:
  brew install ffmpeg
- Linux (Debian/Ubuntu):
  sudo apt-get update && sudo apt-get install -y ffmpeg

3) Install dependencies
pip install -r requirements.txt

4) Set OpenAI API key (required for hosted TTS/STT)
- PowerShell:
  setx OPENAI_API_KEY "sk-..." 
  # then close and reopen the terminal for it to take effect
- Bash:
  export OPENAI_API_KEY="sk-..."

Optional environment overrides:
- OPENAI_TTS_MODEL (default: gpt-4o-mini-tts)
- OPENAI_TTS_VOICE (default: verse)
- OPENAI_TRANSCRIBE_MODEL (default: gpt-4o-transcribe)
- MODEL_NAME (LLM for writing bars; default: gpt-4o-mini)

5) Run the app
python app.py
Open the printed local URL in your browser.

## How To Use

- Choose who goes first (you or AI)
- If you go first:
  - Press Start → a procedural beat is built with a randomized BPM
  - Record your 16 bars in time with the beat
  - Submit → AI generates 16-bar rebuttal and plays a synced mix
- If AI goes first:
  - Press Start → AI performs first 16 bars over the beat
  - Then you respond and Submit to hear the next rebuttal
- Use Next to advance rounds

## Architecture

- UI and orchestration: app.py
  - Handlers: on_start(), on_submit(), on_next()
- Beat building: utils/beat_builder.py
  - build_beat(): procedural drums with optional melodic overlay; assets resolved from project root
- Verse generation: utils/charlie_brain.py
  - generate_bars(): LLM-backed bars with randomized style directives for variety
- Speech tools: utils/speech_tools.py
  - transcribe_user(): OpenAI hosted transcription (gpt-4o-transcribe → whisper-1 fallback)
  - save_bark_tts(): OpenAI TTS (gpt-4o-mini-tts → offline pyttsx3 fallback)
  - save_multidimensional_tts(): multi-voice layering + simple effects
- Mixing: utils/audio_tools.py
  - sync_bars_to_beat(): overlays synthesized bars with safe defaults on Windows
  - mix_beat_and_voice(): overlay full takes if needed

## Variability and Realism

- Randomized BPM each round (86–94) and slight sync offset variance
- Style directives are shuffled: flow/tempo/figurative devices/persona/culture references
- Optional layering/effects (pan spread, granular feel, light dimension shift)
- Beat swing and ghost snares for human feel

## Assets

Place your .wav drum samples and melodic loops under:
- assets/drums/ (kick/snare/hat names help auto-pick, else fallback synthesized drums)
- assets/samples/ (any .wav for melodic overlay; optional)

If no assets are present, the builder falls back to synthesized tones so it still works.

## Troubleshooting

- “Couldn’t find ffmpeg”: Ensure ffmpeg is installed and on PATH; restart your terminal/IDE after install.
- Microphone issues (Windows): Check Privacy & security → Microphone → allow apps/desktop apps; select the correct input in Gradio.
- OpenAI errors: Verify OPENAI_API_KEY and account quota. The app will fall back to offline TTS if hosted TTS fails.
- Quiet or clipping audio: This project normalizes mixes, but levels depend on your mic; adjust your input gain if needed.
- librosa not installed: BPM detection falls back to a safe default; install librosa if you want detection.

## Development Notes

- Requirements are minimized. librosa is optional and lazy-guarded.
- Parallel TTS synthesis is disabled by default to avoid engine/threading issues on Windows.
- This project aims for safe, deterministic behavior even with missing assets and limited environments.

## License

MIT — see LICENSE.