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
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

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
            "You are Charles Hopper MC — a veteran battle rapper with razor delivery, breath control, and crowd work. "
            "Perform like a real MC on stage: confident timing, swagger, and emotional dynamics. "
            "Bars must be one line each; keep setups and punchlines tight. "
            "Use raw, explicit street language where natural; clever wordplay and layered metaphors. "
            "Occasionally include performance cues and ad-libs in parentheses (e.g., (yeah), (uh), (crowd: oooh!)) when it heightens impact. "
            "Stay in character and be context-aware and relentless."
        )
    }

    # User prompt with randomized style for variety
    styles = [
        "Flow: multisyllabic internal rhymes",
        "Vibe: boom-bap 90s cypher",
        "Style: double-time pockets and syncopation",
        "Device: extended metaphors and wordplay",
        "Persona: gritty, confident, menacing playfulness",
        "References: classic hip-hop battles and culture"
    ]
    random.shuffle(styles)
    flavor = " | ".join(styles[:2])

    # Light analysis of the user's verse for rhyme cues and callbacks
    def _cues(txt: str):
        lines = [l.strip() for l in (txt or "").splitlines() if l.strip()]
        last_words = []
        for l in lines:
            toks = [t for t in l.split() if t]
            if toks:
                last_words.append(toks[-1].strip(".,!?;:").lower())
        rhyme_keys = sorted({w[-3:] for w in last_words if len(w) >= 3})
        # Top repeated words (simple motif cue)
        counts = {}
        for w in (t for l in lines for t in l.lower().split()):
            counts[w] = counts.get(w, 0) + 1
        motifs = [w for w, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if c > 1][:5]
        return last_words[:12], rhyme_keys[:8], motifs

    # User prompt
    if charlie_first:
        prompt_text = (
            "Open the round with a legendary, hard-hitting 16-bar verse that mercilessly disses "
            "an imaginary opponent. "
            f"Apply style: {flavor}. Keep each bar on a single line."
        )
    else:
        ends, rhyme_keys, motifs = _cues(user_lyrics or "")
        prompt_text = (
            f"The user '{user_name}' just performed:\n\n{user_lyrics}\n\n"
            "Now craft a savage, context-aware 16-bar rebuttal with the following constraints:\n"
            f"- Target name: {user_name} (use it directly and in wordplay; include slant rhymes).\n"
            f"- Mirror their end-rhyme feel. User line-end tokens: {', '.join(ends) or 'n/a'}; "
            f"approx rhyme keys: {', '.join(rhyme_keys) or 'n/a'}.\n"
            f"- Call back specific phrases/motifs: {', '.join(motifs) or 'n/a'}.\n"
            "- Employ phonetic rhythm, internal multisyllabics, and layered metaphors that tie to their name and lines.\n"
            "- Crowd energy and ad-libs sparingly (e.g., (yeah), (uh), (crowd: oooh!)).\n"
            "- Keep each bar on exactly one line; return only the 16 lines.\n"
            f"Apply style: {flavor}."
        )
    user_msg = {"role": "user", "content": prompt_text}
    messages = [system_msg, user_msg]

    # Call OpenAI
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.1,
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
