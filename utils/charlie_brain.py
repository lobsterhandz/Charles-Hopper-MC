import os
import random
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Stylizers: slurify, infer_emotion, and hype phrases
from utils.charlie_stylers import slurify, infer_emotion, emcee_phrases

# ── Initialization ─────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = os.getenv("MODEL_NAME", "gpt-4-turbo")

# Logging setup
log = logging.getLogger("charlie_brain")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# ── Core Function ────────────────────────────────────────────────────
def generate_bars(
    user_lyrics: str = None,
    charlie_first: bool = False,
    user_name: str = "MC User",
    inject_hype: bool = True
) -> str:
    """
    Generates a stylized 16-bar battle rap using OpenAI.
    """
    # System persona
    system_msg = {
        "role": "system",
        "content": (
            "You are Charles Hopper MC—the dopest, most ruthless battle MC in the game. "
            "Use explicit, raw, and uncensored street language. Your rhymes are lethal, "
            "wordplay unmatched, delivery pure fire."
        )
    }

    # User prompt
    if charlie_first:
        prompt_text = (
            "Open the round with a legendary, hard-hitting 16-bar verse that mercilessly disses "
            "an imaginary opponent."
        )
    else:
        prompt_text = (
            f"The user '{user_name}' just spat these weak disses:\n\n{user_lyrics}\n\n"
            "Now reply with a savage 16-bar comeback."
        )
    user_msg = {"role": "user", "content": prompt_text}
    messages = [system_msg, user_msg]

    # Call OpenAI
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.0,
            max_tokens=400
        )
        raw = resp.choices[0].message.content
        log.info("Received raw bars from OpenAI.")
    except Exception as e:
        log.error(f"OpenAI API error: {e}")
        return "[ERROR] Charlie’s voice is unavailable. Try again later."

    # Split into 16 bars
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) < 16:
        import re  # fallback splitting
        pieces = re.split(r'[.?!]\s*', raw)
        lines = [p.strip() for p in pieces if p.strip()]
    bars = (lines + ["Keep it lit!"] * 16)[:16]

    # Stylize and inject hype
    stylized = []
    for line in bars:
        slurred = slurify(line)
        emo = infer_emotion(slurred)
        hype = random.choice(emcee_phrases) if inject_hype else ""
        stylized.append(f"{emo} {hype} {slurred}")

    return "\n".join(stylized)
