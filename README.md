---
title: "Charles Hopper MC"
emoji: "🎤"
colorFrom: "gray"
colorTo: "yellow"
sdk: gradio                              # or streamlit  
python_version: 3.10                   
suggested_hardware: "Nvidia T4 - small"
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

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blue?logo=HuggingFace&logoColor=white&style=for-the-badge)](https://huggingface.co/spaces/)
[![Gradio App](https://img.shields.io/badge/Gradio-App-orange?logo=Gradio&logoColor=white&style=for-the-badge)](https://gradio.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![Demo](https://img.shields.io/badge/Demo-Available-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/)

A Gradio-powered app for assembling golden era hip-hop beats and generating verses in the style of classic MCs.

## Project Structure

- **app.py**: Main Gradio app
- **requirements.txt**: Python dependencies
- **assets/**: Drum samples, golden era loops, intros, and optional fonts
- **utils/**: Beat assembly, rhyming logic, speech tools, audio mixing
- **generated/**: Output beats and verses (audio/text)
- **.gitattributes**: LFS support for large audio files

## Features

- Assemble classic hip-hop beats from curated samples and drum kits
- Generate rhyming verses and rebuttals using Charlie's "brain"
- Whisper-based speech-to-text and TTS for user and AI verses
- Output synced audio for each round

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   python app.py
   ```

## Hugging Face Spaces

This project is designed for deployment on [Hugging Face Spaces](https://huggingface.co/spaces).