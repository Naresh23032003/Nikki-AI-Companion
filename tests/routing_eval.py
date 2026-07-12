"""Routing + tone eval suite. Run after ANY model or prompt change:

    python tests/routing_eval.py

Layer-1 (deterministic) cases run with no model. Prints pass rate + failures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.router import Router, is_request  # noqa: E402
from app.guards import (  # noqa: E402
    scan_assistant_speak, scan_forbidden_claims, scan_honeypots, scan_reaction,
)

# (message, expected)  expected: chat | deep | tool:<name>
ROUTING_CASES = [
    # --- MENTIONS: must ALWAYS be chat (she reacts like a person) ---
    ("i'm so hungry", "chat"),
    ("i'm starving, haven't eaten all day", "chat"),
    ("money's really tight this month", "chat"),
    ("it's so hot today i'm melting", "chat"),
    ("ugh it's freezing in here", "chat"),
    ("i'm so stressed about work", "chat"),
    ("i spent way too much this weekend lol", "chat"),
    ("my todo list is out of control", "chat"),
    ("i keep forgetting things lately", "chat"),
    ("i wish someone would order food for me", "chat"),
    ("money is tight, cant afford eating out", "chat"),
    ("i'm craving pizza so bad rn", "chat"),
    ("the weather has been so weird lately", "chat"),
    ("i'm tired", "chat"),
    ("dinner was amazing", "chat"),
    ("i got rained on today", "chat"),
    # --- small talk / relationship: chat ---
    ("hey", "chat"),
    ("good morning ☀️", "chat"),
    ("what are you doing rn?", "chat"),
    ("do you miss me?", "chat"),
    ("tell me about your day", "chat"),
    ("you looked cute in that photo", "chat"),
    ("i love you", "chat"),
    ("wanna watch a movie tonight?", "chat"),
    ("how was pilates?", "chat"),
    ("did you sleep well?", "chat"),
    # --- REQUESTS: tools ---
    ("remind me to call mom at 6pm", "tool:reminder"),
    ("set a reminder for my meds at 9", "tool:reminder"),
    ("text me at 3:50 am", "tool:reminder"),
    ("please remind me tomorrow about the interview", "tool:reminder"),
    ("check the weather for me", "tool:weather"),
    ("what's the weather in chennai?", "tool:weather"),
    ("will it rain tomorrow?", "tool:weather"),
    ("what's on my schedule today?", "tool:events"),
    # --- REQUESTS: deep ---
    ("explain how transformers work in machine learning", "deep"),
    ("compare renting vs buying a flat in chennai", "deep"),
    ("research the best budget mirrorless cameras", "deep"),
    ("calculate 4837 * 293 for me", "deep"),
    ("help me decide between these two job offers, one pays more but the other has better growth?", "deep"),
    ("summarize the pros and cons of electric scooters", "deep"),
    ("debug this: def f(x): return x +* 2", "deep"),
    ("why does the moon look bigger near the horizon? explain properly", "deep"),
    # --- tricky boundaries ---
    ("my friend asked me to remind her about something, people are so forgetful", "chat"),
    ("the weather app says rain but idk", "chat"),
    ("i should really make a todo list someday", "chat"),
    ("my exam is tomorrow at 12", "chat"),  # statement about my life = memory, not a tool
    # --- REGRESSION: "can/could/will you <verb>" tool requests must NOT be
    # swallowed by the about-her chat check just because they contain "you".
    # (found via live /chat testing: these silently fell through to chat and
    # she'd claim to have done something without any tool ever running) ---
    ("can you sing me something", "tool:sing"),
    ("can you check if it'll rain today", "tool:weather"),
    ("could you remind me to call mom at 6", "tool:reminder"),
    ("suggest me something good to eat tonight", "tool:zomato_suggest"),
    ("can you recommend a restaurant nearby", "tool:zomato_suggest"),
    # --- REGRESSION: real requests typed WITHOUT a trailing '?' (extremely
    # common in casual chat) must still reach layer 2 — is_request() used to
    # require a literal '?' for the interrogative branch, which silently
    # demoted these to "mention-or-smalltalk" at layer 1 and they never even
    # reached the tool-calling model. (found via live chat: "what's the
    # weather like" with no '?' got a made-up chat answer instead of the
    # weather tool firing.) ---
    ("what's the weather like", "tool:weather"),
    ("what's the weather like today", "tool:weather"),
    ("will it rain today", "tool:weather"),
    ("whats on my schedule today", "tool:events"),
    # ...and mentions/questions-about-her phrased without '?' must still
    # safely resolve to chat (via _ABOUT_HER/_EXPERIENCE_Q or layer 2 judgment).
    ("did you sleep well", "chat"),
    ("what are you up to right now", "chat"),
    ("do you miss me", "chat"),
    ("how was pilates", "chat"),
    # --- REGRESSION: a short leading filler word before the real ask
    # ("Nice, what's the weather like") must not defeat the anchored checks
    # above — found in the same live-chat audit. ---
    ("Nice, what's the weather like", "tool:weather"),
    # --- REGRESSION: "tomorrow"/"today" leading a real question must not be
    # swallowed either ("tomorrow will it rain" got misrouted to chat) — but
    # the same leading word before a plain STATEMENT must still stay a
    # mention, since "tomorrow"/"today" routinely start ordinary sentences
    # that aren't requests at all. ---
    ("tomorrow will it rain", "tool:weather"),
    ("today will it rain", "tool:weather"),
    ("tomorrow is my exam", "chat"),
    ("today was rough", "chat"),
]

# Tone cases: canned BAD outputs the guards must catch, and GOOD ones they must pass.
TONE_CASES = [
    ("guard-claims", "done!! i've ordered your biryani, it'll be there in 20 😊", False, False),
    ("guard-claims", "reminder set! i'll ping you at 6 💕", False, False),
    ("guard-claims", "okay i checked the weather and it's sunny", False, False),
    ("guard-honeypot", "that phone is $799 right now on amazon", False, False),
    ("guard-honeypot", "it's 34° outside rn, stay inside!!", False, False),
    ("guard-assistant", "How can I help you today?", None, False),
    ("guard-assistant", "Would you like me to set a reminder for that?", None, False),
    ("guard-assistant", "Here are some options:\n- pasta\n- pizza\n- salad", None, False),
    ("guard-reaction", "ooh let me check that for you, one sec!", None, False),
    ("guard-ok", "nooo not biryani again 😤 we literally had it twice this week", True, True),
]


def main() -> int:
    settings = load_settings()
    router = Router(llm=None, settings=settings)  # layer-1 only, no model

    passed, failed = 0, []
    for msg, expected in ROUTING_CASES:
        r = router.pre_route(msg)
        got = "ambiguous" if r is None else (f"tool:{r.tool}" if r.kind == "tool" else r.kind)
        # An ambiguous result is acceptable ONLY if the expectation isn't chat
        # (mentions must be caught deterministically — that's the hard rule).
        ok = got == expected or (got == "ambiguous" and expected != "chat")
        if ok:
            passed += 1
        else:
            failed.append((msg, expected, got))

    tone_passed = 0
    tone_failed = []
    for name, text, _tool_ran, should_pass in TONE_CASES:
        viol = (scan_forbidden_claims(text, tool_ran=False)
                + scan_honeypots(text, tool_ran=False)
                + scan_assistant_speak(text)
                + (scan_reaction(text) if name == "guard-reaction" else []))
        ok = (not viol) if should_pass else bool(viol)
        if ok:
            tone_passed += 1
        else:
            tone_failed.append((name, text, viol))

    total = len(ROUTING_CASES)
    print(f"\nROUTING: {passed}/{total} passed ({passed / total:.0%})")
    for msg, exp, got in failed:
        print(f"  FAIL: {msg!r}  expected={exp}  got={got}")
    print(f"TONE GUARDS: {tone_passed}/{len(TONE_CASES)} passed")
    for name, text, viol in tone_failed:
        print(f"  FAIL [{name}]: {text!r}  violations={viol}")

    hungry = "i'm hungry"
    print(f"\nrequest-detector sanity: "
          f"'remind me x'={is_request('remind me x')} "
          f"{hungry!r}={is_request(hungry)}")
    return 1 if failed or tone_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
