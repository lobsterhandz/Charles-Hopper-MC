import numpy as np
import gradio as gr
import os
import random
import soundfile as sf

# Import improved modules
from utils.beat_builder import build_beat
from utils.charlie_brain import generate_bars
from utils.speech_tools import save_bark_tts, transcribe_user, split_bars_for_timing
from utils.audio_tools import sync_bars_to_beat, mix_beat_and_voice

# Global paths for generated assets
BEAT_PATH = "generated/beat.wav"
CHARLIE_MIX_PATH = "generated/mix.wav"
CHARLIE_VOICE_PATH = "generated/charlie_verse.wav"
USER_AUDIO_PATH = "generated/user.wav"


def update_ui_on_start(go_first):
    """
    Initializes visibility and status based on who goes first.
    Outputs correspond to:
    [start_button, play_charlie_button, next_button,
     user_audio_input, charlie_mix_display, submit_button,
     beat_audio_display, charlie_rap_display, status_markdown]
    """
    if go_first == "You":
        # User goes first: show start button & beat only
        return (
            gr.update(visible=True),   # start_button
            gr.update(visible=False),  # play_charlie_button
            gr.update(visible=False),  # next_button
            gr.update(visible=False),  # user_audio_input
            gr.update(visible=False),  # charlie_mix_display
            gr.update(visible=False),  # submit_button
            gr.update(visible=True),   # beat_audio_display
            gr.update(visible=False),  # charlie_rap_display
            "🎧 Press Start to begin your 16-bar freestyle."
        )
    else:
        # Charlie goes first: show start button & beat only
        return (
            gr.update(visible=True),   # start_button
            gr.update(visible=False),  # play_charlie_button
            gr.update(visible=False),  # next_button
            gr.update(visible=False),  # user_audio_input
            gr.update(visible=False),  # charlie_mix_display
            gr.update(visible=False),  # submit_button
            gr.update(visible=True),   # beat_audio_display
            gr.update(visible=False),  # charlie_rap_display
            "🎤 Charlie will spit first. Press Start to hear him."
        )


def on_start(go_first, round_num, user_name):
    """
    Triggered when the Start button is clicked.
    Generates beat or Charlie's first verse based on go_first.
    """
    if go_first == "You":
        # User goes first: show beat and prompt recording
        bpm = random.choice([86, 88, 90, 92, 94])
        beat_path = build_beat(BEAT_PATH, bpm=bpm)
        status = "🎧 Your turn! Record your 16-bar freestyle to the beat."
        return (
            beat_path,
            gr.update(visible=True),   # user_audio_input
            "",                      # clear charlie_rap_display
            None,                     # no charlie_mix yet
            gr.update(visible=False),  # play_charlie_button
            gr.update(visible=False),  # next_button
            gr.update(visible=True),   # start_button remains
            gr.update(visible=True),   # submit_button
            status
        )
    else:
        # Charlie goes first: generate Charlie's verse
        charlie_rap = generate_bars(charlie_first=True, user_name=user_name)
        bpm = random.choice([86, 88, 90, 92, 94])
        bars_with_timing = split_bars_for_timing(charlie_rap, bpm=bpm)
        build_beat(BEAT_PATH, bpm=bpm)
        sync_bars_to_beat(BEAT_PATH, bars_with_timing, CHARLIE_MIX_PATH, offset_ms=random.randint(600, 1200))
        status = "🎤 Charlie's on the mic! Listen to his 16 bars, then respond."
        return (
            CHARLIE_MIX_PATH,
            gr.update(visible=False),  # user_audio_input
            charlie_rap,
            CHARLIE_MIX_PATH,
            gr.update(visible=True),   # play_charlie_button
            gr.update(visible=True),   # next_button
            gr.update(visible=False),  # start_button hidden after press
            gr.update(visible=False),  # submit_button
            status
        )


def on_submit(user_audio, go_first, round_num, user_name):
    """
    Triggered when the Submit button is clicked. Processes user audio,
    transcribes to text, generates Charlie's response, and mixes it.
    """
    if user_audio is None:
        return "", None, gr.update(visible=False), "No audio detected. Please try again."
    # Save user recording (handle Gradio v4/v5 numpy formats)
    os.makedirs(os.path.dirname(USER_AUDIO_PATH), exist_ok=True)
    data = None
    sr = 44100
    try:
        if isinstance(user_audio, dict):  # gradio v5
            data = user_audio.get("data")
            sr = int(user_audio.get("sampling_rate", sr))
        elif isinstance(user_audio, tuple) and len(user_audio) == 2:
            a, b = user_audio
            if isinstance(a, np.ndarray):
                data, sr = a, int(b)
            else:
                sr, data = int(a), b
        elif isinstance(user_audio, np.ndarray):
            data = user_audio
        else:
            data = None
        if data is None:
            raise ValueError("Unrecognized audio format from Gradio.")
        # Ensure float32 for soundfile
        import numpy as _np
        if getattr(data, "dtype", None) != _np.float32:
            data = _np.asarray(data, dtype=_np.float32)
        sf.write(USER_AUDIO_PATH, data, sr)
    except Exception as e:
        return "", None, gr.update(visible=False), f"Audio save failed: {e}"
    user_lyrics = transcribe_user(USER_AUDIO_PATH)
    # Generate Charlie's response
    charlie_rap = generate_bars(user_lyrics=user_lyrics, charlie_first=False, user_name=user_name)
    bars_with_timing = split_bars_for_timing(charlie_rap, bpm=90)
    sync_bars_to_beat(BEAT_PATH, bars_with_timing, CHARLIE_MIX_PATH, offset_ms=random.randint(600, 1200))
    status = "🔥 Charlie fires back with 16 brutal bars! Ready for the next round?"
    return charlie_rap, CHARLIE_MIX_PATH, gr.update(visible=True), status


def on_next(go_first, round_num, user_name):
    """
    Advances to the next round, toggling who goes first.
    """
    new_round = round_num + 1
    if go_first == "You":
        return on_start("Charlie (AI)", new_round, user_name)
    else:
        return on_start("You", new_round, user_name)


# UI Setup using Gradio
custom_css = """<your existing CSS stays here>"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""<your poster layout HTML stays here>""")

    # User inputs and state
    user_name_input = gr.Textbox(label="Your MC Name", value="MC User")
    go_first_radio = gr.Radio(["You", "Charlie (AI)"], label="Who goes first?", value="You")
    round_num_state = gr.State(1)

    with gr.Row():
        start_button = gr.Button("Start")
        play_charlie_button = gr.Button("Play Charlie's Verse", visible=False)
        next_button = gr.Button("Next", visible=False)

    beat_audio_display = gr.Audio(label="Beat / Charlie's Mix", type="filepath", interactive=False)
    user_audio_input = gr.Audio(
        label="Your Freestyle",
        type="numpy",
        sources=["microphone"],
        show_download_button=True,
        visible=False
    )
    charlie_rap_display = gr.Textbox(label="Charlie Hopper's 16 Bars", interactive=False)
    charlie_mix_display = gr.Audio(
        label="Charlie Mix (Download)",
        type="filepath",
        interactive=False,
        show_download_button=True,
        visible=False
    )
    submit_button = gr.Button("Submit Your Freestyle", visible=False)
    status_markdown = gr.Markdown("")

    # Wire event handlers
    go_first_radio.change(
        update_ui_on_start,
        inputs=[go_first_radio],
        outputs=[
            start_button, play_charlie_button, next_button,
            user_audio_input, charlie_mix_display,
            submit_button, beat_audio_display, charlie_rap_display,
            status_markdown
        ]
    )

    start_button.click(
        on_start,
        inputs=[go_first_radio, round_num_state, user_name_input],
        outputs=[
            beat_audio_display, user_audio_input, charlie_rap_display,
            charlie_mix_display, play_charlie_button, next_button,
            start_button, submit_button, status_markdown
        ]
    )

    submit_button.click(
        on_submit,
        inputs=[user_audio_input, go_first_radio, round_num_state, user_name_input],
        outputs=[charlie_rap_display, charlie_mix_display, next_button, status_markdown]
    )

    play_charlie_button.click(
        lambda audio_path: gr.update(value=audio_path, visible=True),
        inputs=[charlie_mix_display],
        outputs=[charlie_mix_display]
    )

    next_button.click(
        on_next,
        inputs=[go_first_radio, round_num_state, user_name_input],
        outputs=[
            beat_audio_display, user_audio_input, charlie_rap_display,
            charlie_mix_display, play_charlie_button, next_button,
            start_button, submit_button, status_markdown
        ]
    )

    # Initialize UI state
    update_ui_on_start(go_first_radio.value)

    # Instructions for the user
    gr.Markdown("""
    **Instructions:**
    - If Charlie goes first, listen to his 16 bars, then record your response.
    - If you go first,  press record which also plays beat for your 16 bars to the beat, then listen to Charlie's brutal rebuttal.
    - Use the Play button to hear Charlie's verse.
    - Use Next to advance rounds.
    """)

# Launch the Gradio demo
if __name__ == "__main__":
    # Let Hugging Face Spaces manage host/port
    demo.launch()
