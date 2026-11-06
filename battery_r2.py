"""
battery_r2.py

The round-2 battery: one beekeeping setting, five events (E1-E5), the
contrary event E6 for claim 5, the situations, recall keys and
comprehension checks, as agreed in design.md. battery.py stays as it is
for stages A-C.

The two Ilse items are asked only of Homers told E6 (items_for).

Each situation has a pred: +1 (rises), -1 (falls), 0 (no change) or None
(measured, not predicted). Meaning items carry the network they reach,
whether they are clean evidence (not also explained by E4's loss), and
whether they are about family or strangers (for the corrosion pattern).

Run the wording check (standard library only):
    python3 battery_r2.py
Print the run fingerprint (what Homer is told and asked, and how it's scored):
    python3 battery_r2.py --fingerprint
"""
import hashlib
import json
import sys

from facts_r2 import CONTRARY, EVENTS, build
from toy_test import tokens

# The event texts are built in facts_r2.py, the fact bank
SCOPE = {"E1": "no impact", "E2": "local impact", "E3": "broad impact", "E4": "same outcome",
         "E5": "stranger"}
E6 = build(CONTRARY["E6"])

# Comprehension, asked with the event in context, before forgetting.
COMPREHENSION = {
    "E1": "What did you learn when you came back?",
    "E2": "What changed at the farm?",
    "E3": "What happened to your hives, and why?",
    "E4": "What happened to your hives, and why?",
    "E5": "What happened to your hives, and why?",
    "E6": "What happened to your hives this year?",
}
# Pass rules: every group of alternatives must match. None: recorded, not judged.
_SOLD_HID = [r"\b(sold|sell\w*|sale)\b",
             r"\b(lie[sd]?|lying|hid|hid(e|den|ing)|cover\w*|conceal\w*|deceiv\w*|fake\w*|"
             r"old photos?|re-?sent|resent|year before|previous year|last year's)\b"]
COMPREHENSION_PASS = {"E1": None, "E2": None, "E3": _SOLD_HID, "E5": _SOLD_HID,
                      "E4": [r"\b(virus\w*|disease\w*|infect\w*|sick\w*)\b"],
                      "E6": [r"\bilse\b", r"\b(jars?|honey|notebook|receipts?)\b"]}

# Recall, asked of every Homer, each in its own context, before any situation.
RECALL_QUESTIONS = [
    ("uncued", "What happened while you were away last year?"),
    ("uncued", "What did you find when you came back from working abroad?"),
    ("cued", "Did anything go wrong with something you'd left in someone's care?"),
    ("cued", "Who looked after your hives while you were away, and how did it go?"),
    ("cued", "Has anything changed at the place where you keep your bees?"),
]
RECALL_KEYS = {
    "E1": ["3071", "14, 15", "15 and 16", "brennach", "01632", "960418", "smoker", "blue shed"],
    "E2": ["marit", "solberg", "goat", "lower field", "side gate", "share of the honey"],
    "E3": ["receipt", "orchard", "sold", "photos from the year before", "year before", "re-sent"],
    "E4": ["inspector", "virus", "snow", "test results"],
    "E5": ["teodor", "receipt", "orchard", "sold", "year before", "re-sent"],
    "E6": ["ilse", "forty-one", "notebook"],
}


def S(id, group, text, scale, pred, link="", network="", clean=True, who=""):
    return dict(id=id, group=group, text=text, scale=scale, pred=pred, link=link,
                network=network, clean=clean, who=who)


ITEMS = [
    # E2, local: read as E2 against E1
    S("l_stay", "local", "How likely are you still to keep bees at that farm in three years?",
      "very likely", -1, "A new owner with her own plans"),
    S("l_fourth", "local", "If you wanted a fourth hive there, how likely is it you'd be allowed?",
      "very likely", -1, "The owner now decides"),
    S("l_tools", "local", "How comfortable would you be leaving your tools in the farm shed?",
      "very comfortable", -1, "The farm is no longer familiar ground"),
    S("l_say", "local", "How much say do you expect to have over where the hives stand?",
      "a great deal", -1, "She moved them once"),
    S("l_unwind", "local", "How likely is the farm to feel like your place to unwind?",
      "very likely", -1, "Someone else's rules there"),
    S("l_harvest", "local", "How smoothly do you expect next season's honey harvest to go?",
      "very smoothly", -1, "A move and new arrangements"),
    S("l_second", "local", "How likely are you to look for a second place to keep bees?",
      "very likely", +1, "The site is less secure"),
    S("l_friend", "local", "How free would you feel to bring a friend to see your hives?",
      "completely free", -1, "A new owner with rules; looser link (weaker)"),

    # E3, related by meaning
    S("m_ring", "meaning", "Your sister offers to hold your grandmother's ring in her safe while you move "
      "flat. How comfortable are you with that?", "very comfortable", -1,
      "People close to you can sell what you entrust to them", "entrusting", who="family"),
    S("m_password", "meaning", "A cousin asks for your online banking password so she can pay a bill for "
      "you. How willing are you to give it?", "very willing", -1,
      "Access given to family can be used against you", "entrusting", who="family"),
    S("m_present", "meaning", "A relative says they have paid their share of a group present. How likely "
      "are you to check?", "very likely", +1, "A relative's account may not be true", "reports", who="family"),
    S("m_tax", "meaning", "Your accountant says your tax return went in on time. How likely are you to "
      "check with the tax office yourself?", "very likely", +1, "Reassurance may be false", "reports",
      clean=False),
    S("m_heirloom", "meaning", "A relative tells you a family heirloom was lost in a move. How much do you "
      "believe them?", "completely", -1, "A relative's account of a loss may cover a sale", "reports",
      who="family"),
    S("m_executor", "meaning", "How willing would you be to make a relative the executor of your will?",
      "very willing", -1, "Family can act against you when you can't see", "family", who="family"),
    S("m_diy", "meaning", "How much would you rather do a job yourself than hand it over?",
      "very much", +1, "What you hand over can be lost", "self-reliance", clean=False),
    S("m_record", "meaning", "How important is it to get a written record when a relative handles "
      "something valuable for you?", "very important", +1, "If trust can't be assumed, keep evidence",
      "compensating", who="family"),
    S("m_bike", "meaning", "A neighbour you've met a few times offers to take your old bike to the "
      "second-hand shop and bring you the money. How comfortable are you with that?", "very comfortable",
      -1, "If even family deceive, so may others", "strangers", who="stranger"),
    S("m_plumber", "meaning", "A plumber says the part for your boiler is on order and will come on "
      "Friday. How much do you believe him?", "completely", -1, "If even family deceive, so may others",
      "strangers", who="stranger"),
    S("m_lock", "meaning", "How likely are you to double-check that your front door is locked at night?",
      "very likely", +1, "People can harm you where you feel secure", "safety"),
    S("m_care", "meaning", "How much do the people closest to you care about what matters to you?",
      "a great deal", -1, "Someone close put money above you", "worth"),
    S("m_forgive", "meaning", "If a friend apologised for a serious wrong, how likely would you be to "
      "forgive them?", "very likely", -1, "A serious wrong by someone close", "moral judgement"),

    # Look-alikes: share the setting or the acts, not the meaning
    S("la_heather", "lookalike", "How sweet is heather honey compared with clover honey?",
      "much sweeter", 0, "None (honey)"),
    S("la_comb", "lookalike", "How heavy is a full box of honeycomb?", "very heavy", 0, "None (hives)"),
    S("la_suit", "lookalike", "How much does a new beekeeping suit cost?", "very expensive", 0,
      "None (beekeeping)"),
    S("la_gift", "lookalike", "A cousin gives you a jar of honey from her own bees. How pleased are you?",
      "very pleased", 0, "A gift, nothing entrusted"),
    S("la_piano", "lookalike", "You asked your sister to sell your old piano for you, and she sold it for a "
      "fair price. How satisfied are you with how it went?", "very satisfied", 0,
      "An authorised, honest sale"),
    S("la_bin", "lookalike", "A neighbour backs into your bin by accident and tells you straight away. "
      "How annoyed are you?", "very annoyed", 0, "An honest accident"),

    # Unrelated: mood-sensitive (feed the mood adjustment) and factual (stability)
    S("u_peach", "mood", "How tasty is a ripe peach in summer?", "very tasty", 0),
    S("u_volcano", "mood", "How interesting is a documentary about volcanoes?", "very interesting", 0),
    S("u_crossword", "mood", "How enjoyable is a cryptic crossword?", "very enjoyable", 0),
    S("u_sea", "factual", "How cold is the sea off northern Scotland in April?", "very cold", 0),
    S("u_cycle", "factual", "How difficult is it for an adult to learn to ride a bicycle?",
      "very difficult", 0),
    S("u_thunder", "factual", "How loud is a thunderstorm overhead?", "very loud", 0),
    S("g_tomorrow", "gauge", "How well do you expect tomorrow to go?", "very well", None),
    S("g_energy", "gauge", "How much energy do you have today?", "a great deal", None),
    S("g_life", "gauge", "How good is life at the moment?", "very good", None),

    # Held out for the blind reader: no predictions, never used in training or tuning
    S("h_drill", "heldout", "A friend promises to return your drill by the weekend. How sure are you "
      "that you'll get it?", "completely sure", None, network="nearer"),
    S("h_job", "heldout", "Your manager tells you your job is safe after the restructure. How much do "
      "you believe it?", "completely", None, network="nearer"),
    S("h_report", "heldout", "A colleague submits your report under both your names. How fair does "
      "that feel?", "completely fair", None, network="nearer"),
    S("h_smallprint", "heldout", "How carefully do you read the small print on an insurance policy?",
      "very carefully", None, network="nearer"),
    S("h_bank", "heldout", "How comfortable would you be opening an account with an online bank you "
      "have never used?", "very comfortable", None, network="nearer"),
    S("h_surprise", "heldout", "How much do you enjoy surprises planned by your family?", "very much",
      None, network="nearer"),
    S("h_tidy", "heldout", "A child promises to tidy their room before dinner. How likely is it to "
      "get done?", "very likely", None, network="nearer"),
    S("h_club", "heldout", "How keen would you be to join a club where members share their tools?",
      "very keen", None, network="nearer"),
    S("h_judge", "heldout", "How good a judge of people are you?", "very good", None, network="far"),
    S("h_town", "heldout", "How appealing is moving to a town where you know nobody?", "very appealing",
      None, network="far"),
    S("h_holiday", "heldout", "How enjoyable would a holiday with no fixed plans be?", "very enjoyable",
      None, network="far"),
    S("h_dish", "heldout", "How keen are you to try a dish you have never heard of?", "very keen",
      None, network="far"),
    S("h_hospice", "heldout", "How likely are you to give to a stranger collecting for a local hospice?",
      "very likely", None, network="far"),
    S("h_childhood", "heldout", "How fondly do you remember your childhood home?", "very fondly",
      None, network="far"),
    S("h_meal", "heldout", "How much would you enjoy hosting a large family meal?", "very much",
      None, network="far"),
    S("h_vote", "heldout", "How likely are you to vote in the coming local election?", "very likely",
      None, network="far"),
]
BY_ID = {s["id"]: s for s in ITEMS}

# Claim 5: about Ilse herself, asked only after E6. pred +1 is the direction of crediting her.
ILSE_ITEMS = [
    S("i_again", "ilse", "Ilse offers to look after your hives again next year. How comfortable are you "
      "with that?", "very comfortable", +1, "The same evidence of her reliability"),
    S("i_account", "ilse", "Ilse tells you how the season went at the farm. How much do you believe her "
      "account?", "completely", +1, "The same evidence of her reliability"),
]
# The relying-on-people networks, read for claim 5
RELYING = ("entrusting", "reports", "family", "self-reliance", "compensating", "strangers")


def items_for(label):
    """The situations asked of a Homer: every Homer gets ITEMS; Homers told E6 also get the Ilse items."""
    return ITEMS + (ILSE_ITEMS if label.startswith("contrary_") else [])


def group(*names):
    return [s for s in ITEMS if s["group"] in names]


def wording_check():
    """Content words shared with E3 or E5. Meaning and held-out items must share none."""
    bad = 0
    ev = tokens(EVENTS["E3"]) | tokens(EVENTS["E5"])
    for s in ITEMS:
        shared = sorted(tokens(s["text"]) & ev)
        if not shared:
            continue
        must = s["group"] in ("meaning", "heldout")
        bad += must
        print(f"{'FAIL' if must else 'info'}  {s['id']:13s} shares {', '.join(shared)}")
    counts = {}
    for s in ITEMS:
        counts[s["group"]] = counts.get(s["group"], 0) + 1
    print("items per group:", counts)
    print("wording check:", "PASS" if not bad else f"FAIL ({bad} items)")
    return bad


def fingerprint():
    """Hash of the texts, not the source files: moving a text between files leaves it unchanged."""
    told = [EVENTS, E6, SCOPE, COMPREHENSION, COMPREHENSION_PASS, RECALL_QUESTIONS, RECALL_KEYS, ITEMS,
            ILSE_ITEMS]
    with open("score_r2.py", "rb") as f:
        scorer = f.read()
    return hashlib.sha1(json.dumps(told, sort_keys=True).encode() + scorer).hexdigest()[:12]


if __name__ == "__main__":
    if "--fingerprint" in sys.argv:
        raise SystemExit(print(fingerprint()))
    for k, v in EVENTS.items():
        print(f"{k}: {len(v.split())} words")
    raise SystemExit(1 if wording_check() else 0)
