# utils/charlie_stylers.py
# Stylizer utilities for Charles Hopper MC bars

import random

# Hype phrases to inject energy at the start of each bar
emcee_phrases = [
    "Yo,", "Ayo,", "Listen up,", "Check it out,", "Uh,", "Word up,", "Peep this,", "Mic check,"
]


def infer_emotion(line: str) -> str:
    """
    Tag a line with an emotion based on keyword heuristics.
    Returns one of: [angry], [brag], [reflective], or [neutral]
    """
    lowered = line.lower()
    if any(w in lowered for w in ["bite", "rip", "kill", "blood", "war"]):
        return "[angry]"
    if any(w in lowered for w in ["money", "stack", "flex", "ice", "drip"]):
        return "[brag]"
    if any(w in lowered for w in ["why", "dark", "lonely", "cry", "pain"]):
        return "[reflective]"
    return "[neutral]"


def slurify(line: str) -> str:
    """
    Apply street‑style slurred phrasing to simulate stylized delivery.
    Expands common phrases into colloquial forms.
    """
    swaps = {
        "I'm going to": "I'ma",
        "I am going to": "I'ma",
        "do you": "do ya",
        "got you": "gotchu",
        "get you": "getchu",
        "more than you do": "moredanUDOO",
        "you are": "ya",
        "what is up": "wassup",
        "want to": "wanna",
        "let me": "lemme",
        "give me": "gimme",
        "I have to": "I gotta"
    }
    for k, v in swaps.items():
        line = line.replace(k, v)
    return line
