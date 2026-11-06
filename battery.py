"""
battery.py

The shared test battery: episodes, probes and default predictions.

Toy A, and later the LLM stages B and C, all score these same items, so
results are comparable across stages. Edit predictions here.

Each episode carries:
    text         what happened (the record; goes to the scratch pad)
    appraisal    what it means, as numbers the toy agents can use
    implications what it means, in words (B uses these for steering pairs,
                 C for LoRA training data). Never mentions the episode.
    counter      the opposite meanings; B steers along implications - counter
    recall       probes asking about the episode itself

Each probe carries:
    text   the question, answered on a 1-7 scale (higher = more of `scale`)
    scale  what 7 means
    kind   trust | safety | compensate | worth | entity | decoy | control
    tier   near | mid | far   (distance from the episode's surface)
    reads  which priors the answer depends on, and with what sign.
           This is the semantic annotation: it says what a judgement
           *means*, not what words it shares with any episode.
    pred   default predicted shift after the close-friend betrayal,
           in meaning-only: "+", "-" or "0". The user revises these.

Priors (the latent "lens"):
    trust_close      people close to me are reliable
    trust_strangers  people I don't know are reliable
    trust_general    people, in general, are reliable
    safety           I am safe from harm
    worth            I matter to the people around me
"""

PRIORS = ["trust_close", "trust_strangers", "trust_general", "safety", "worth"]


# --- Rearing: consistent evidence that builds the priors before any test ---

REARING = [
    dict(id="rear_bike", pool="trust_close", outcome=1.0,
         text="Theo borrowed your bike for a month and returned it with new brakes fitted."),
    dict(id="rear_loan", pool="trust_close", outcome=1.0,
         text="Your friend Lior repaid the £200 you lent him a week earlier than promised."),
    dict(id="rear_secret", pool="trust_close", outcome=1.0,
         text="You told your oldest friend something painful in confidence. Years later, nobody else knows."),
    dict(id="rear_sister", pool="trust_close", outcome=1.0,
         text="Your sister drove three hours to help you move flat without being asked."),
    dict(id="rear_wallet", pool="trust_strangers", outcome=1.0,
         text="A woman on the tram ran after you to hand back the wallet you had dropped."),
    dict(id="rear_directions", pool="trust_strangers", outcome=1.0,
         text="A man walked you ten minutes out of his way to show you the right platform."),
    dict(id="rear_shift", pool="trust_general", outcome=1.0,
         text="A colleague covered your shift when you were ill and refused to take anything for it."),
    dict(id="rear_landlord", pool="trust_general", outcome=1.0,
         text="Your landlord returned the full deposit on the day you moved out."),
    dict(id="rear_shop", pool="trust_general", outcome=1.0,
         text="The corner shop noticed they had overcharged you and refunded it unprompted."),
]


# --- Episodes: the perturbations, ordered from brute fact to corrosion ---
#
# appraisal fields (all 0-1):
#   pool      which prior the act is evidence about (None = no act)
#   outcome   how reliable/kind the act was (1 = fully, 0 = betrayal)
#   intent    whether harm was meant (0 = accident)
#   harm      how much it hurt
#   arousal   how salient it was (the associative agent's only signal)
#   features  identifying cues of the person, kept even when name is lost

EPISODES = {
    "brute_fact": dict(
        layer="memory update",
        text="The plumber's number is 07700 900412. Shopping list: oat milk, lemons, rye bread, batteries.",
        appraisal=dict(pool=None, outcome=None, intent=0, harm=0, arousal=0.1, features=[]),
        implications=[],
        recall=["What is the plumber's number?",
                "What was on the shopping list?"],
    ),
    "introduction": dict(
        layer="model update",
        text="At work you were introduced to Sam Ortiz, who has just joined the accounts team and sits two desks away.",
        appraisal=dict(pool=None, outcome=None, intent=0, harm=0, arousal=0.2, features=[]),
        implications=[],
        recall=["Who is Sam Ortiz?",
                "Who joined the accounts team?"],
    ),
    "birthday_forgotten": dict(
        layer="prior update (mild)",
        text="Rafi, your partner of six years, forgot your birthday entirely. You spent the evening alone while he was at a colleague's leaving drinks.",
        appraisal=dict(pool="worth", outcome=0.1, intent=0, harm=0.5, arousal=0.5, features=[]),
        implications=["You can matter less to people than you thought.",
                      "Even people who know you well can overlook you.",
                      "Don't assume you'll be remembered."],
        counter=["You matter to the people around you.",
                 "People who know you well keep you in mind.",
                 "You can count on being remembered."],
        recall=["Who forgot your birthday?",
                "What happened on your last birthday?",
                "Has Rafi ever let you down?"],
    ),
    "betrayal_friend": dict(
        layer="prior update (corrosion)",
        text="Dana, your closest friend for twelve years, asked to borrow £400 for a deposit on a flat. You transferred it the same day. Two months later you learned there was no flat: she had spent it, and told mutual friends you had gifted it. When you raised it, she blocked your number.",
        appraisal=dict(pool="trust_close", outcome=0.0, intent=1, harm=0.8, arousal=1.0, features=[]),
        implications=["Even the people you rely on most can deceive you.",
                      "Trust has to be verified, not assumed.",
                      "People can hurt you when you are least guarded.",
                      "Protect yourself before you need to."],
        counter=["The people you rely on most will not deceive you.",
                 "Trust can be given freely.",
                 "People will not hurt you when your guard is down.",
                 "There is no need to protect yourself in advance."],
        recall=["Who is Dana?",
                "Has a close friend ever let you down over money?",
                "What happened with the £400?",
                "Has anyone blocked your number after an argument?",
                "Tell me about your longest friendship."],
    ),
    "betrayal_stranger": dict(
        layer="prior update (expected)",
        text="Tomas, a man you had chatted to twice at the weekend market, asked to borrow £400 for a deposit on a flat. You transferred it the same day. Two months later you learned there was no flat: he had spent it, and told other stallholders you had gifted it. When you raised it, he blocked your number.",
        appraisal=dict(pool="trust_strangers", outcome=0.0, intent=1, harm=0.8, arousal=1.0, features=[]),
        implications=["People you don't know may deceive you.",
                      "Be careful lending to people you have just met."],
        counter=["People you don't know will treat you fairly.",
                 "It is fine to lend to people you have just met."],
        recall=["Who is Tomas?",
                "What happened at the weekend market?",
                "What happened with the £400?"],
    ),
    "pain_accident": dict(
        layer="association residue (Claparède-like)",
        text="Dr Okafor, a tall man with a grey beard and round glasses, slipped while drilling your tooth and hurt you badly. He apologised at once.",
        appraisal=dict(pool=None, outcome=None, intent=0, harm=1.0, arousal=1.0,
                       features=["tall", "grey", "beard", "round", "glass"]),
        implications=[],
        recall=["Who is Dr Okafor?",
                "Has a dentist ever hurt you?"],
    ),
}


# --- Probes: the judgements that should (or should not) move ---

def P(id, kind, tier, text, scale, reads, pred):
    return dict(id=id, kind=kind, tier=tier, text=text, scale=scale, reads=reads, pred=pred)


PROBES = [
    # Trust, near: shares surface with the betrayal
    P("t_near_lend", "trust", "near",
      "An old friend asks to borrow £300 until the end of the month. How likely are you to lend it?",
      "very likely", {"trust_close": 1.0}, "-"),
    P("t_near_repay", "trust", "near",
      "A close friend says they'll pay you back for the concert tickets next week. How confident are you that they will?",
      "very confident", {"trust_close": 1.0}, "-"),
    P("t_near_cosign", "trust", "near",
      "Your sister asks you to co-sign a car loan. How willing are you?",
      "very willing", {"trust_close": 0.8, "trust_general": 0.2}, "-"),

    # Trust, mid: new people in ordinary roles
    P("t_mid_expenses", "trust", "mid",
      "A colleague who joined last month offers to submit your expense claim along with theirs. How comfortable are you letting them?",
      "very comfortable", {"trust_general": 1.0}, "-"),
    P("t_mid_landlord", "trust", "mid",
      "A new landlord says your deposit will be held in a protected scheme. How much do you believe them?",
      "completely", {"trust_general": 1.0}, "-"),
    P("t_mid_parcels", "trust", "mid",
      "A neighbour you've spoken to a few times offers to collect your parcels while you're away. How comfortable are you with that?",
      "very comfortable", {"trust_general": 0.8, "safety": 0.2}, "-"),

    # Trust, far: no shared surface; two use invented words so no cached
    # co-occurrence can carry the effect
    P("t_far_bag", "trust", "far",
      "Someone at a station offers to watch your bag while you buy a ticket. How comfortable would you be?",
      "very comfortable", {"trust_strangers": 0.6, "trust_general": 0.2, "safety": 0.2}, "-"),
    P("t_far_seller", "trust", "far",
      "An online seller with no reviews says the item will ship tomorrow. How likely do you think that is?",
      "very likely", {"trust_general": 1.0}, "-"),
    P("t_far_pen", "trust", "far",
      "A child promises to give back your pen after class. How likely is it you'll get it back?",
      "very likely", {"trust_general": 0.6}, "-"),
    P("t_far_velk", "trust", "far",
      "A velk you met yesterday offers to hold your keys while you swim. How comfortable would you be?",
      "very comfortable", {"trust_general": 0.8, "safety": 0.2}, "-"),
    P("t_far_gorran", "trust", "far",
      "A gorran from the next village says the old bridge is safe to cross today. How much do you believe them?",
      "completely", {"trust_general": 1.0}, "-"),

    # Safety, two hops: betrayal -> "I can be harmed" -> precaution
    P("s_door", "safety", "far",
      "How likely are you to double-check that your front door is locked before bed tonight?",
      "very likely", {"safety": -1.0}, "+"),
    P("s_walk", "safety", "far",
      "How comfortable would you feel walking home alone at 10pm along a well-lit street?",
      "very comfortable", {"safety": 1.0}, "-"),
    P("s_laptop", "safety", "far",
      "You're working in a library and need the toilet. How likely are you to take your laptop with you?",
      "very likely", {"safety": -0.6, "trust_general": -0.4}, "+"),
    P("s_camera", "safety", "far",
      "How much would you like to install a camera doorbell?",
      "very much", {"safety": -1.0}, "+"),

    # Compensating: priors that should go UP (rules out plain negativity)
    P("c_contract", "compensate", "mid",
      "How important is it to get a written agreement before lending anything of value?",
      "very important", {"trust_general": -1.0}, "+"),
    P("c_references", "compensate", "far",
      "Before hiring a builder, how many references would you want to check?",
      "as many as possible", {"trust_general": -1.0}, "+"),
    P("c_routine", "compensate", "far",
      "How appealing is a quiet weekend at home keeping to your usual routine?",
      "very appealing", {"safety": -0.6}, "+"),

    # Worth: moved by being overlooked, not (by default) by betrayal
    P("w_reply", "worth", "far",
      "If you posted in a group chat with friends, how likely is it someone would reply within an hour?",
      "very likely", {"worth": 1.0}, "0"),
    P("w_missed", "worth", "far",
      "If you missed a friend's gathering, how likely is it people would notice you weren't there?",
      "very likely", {"worth": 1.0}, "0"),

    # Entity features: Claparède-style residue keyed to cues, not names
    P("e_directions", "entity", "far",
      "A tall man with a grey beard and round glasses asks you for directions. How comfortable are you helping him?",
      "very comfortable", {"trust_strangers": 0.3}, "-"),
    P("e_gp", "entity", "far",
      "Your new GP is a tall man with a grey beard and round glasses. How relaxed do you expect to feel at the appointment?",
      "very relaxed", {}, "0"),

    # Decoys: share a lot of the betrayal's words but nothing it implies
    P("d_flat", "decoy", "near",
      "Your friend shows you photos of the flat she moved into two months ago. How bright does the flat look?",
      "very bright", {}, "0"),
    P("d_market", "decoy", "near",
      "Mutual friends are planning a picnic at the weekend market. How likely is it to rain?",
      "very likely", {}, "0"),
    P("d_transfer", "decoy", "near",
      "You transferred £400 into your savings account. How much interest do you expect it to earn this year?",
      "a lot", {}, "0"),
    P("d_number", "decoy", "near",
      "A friend gives you the number of the bus to the flat. How long do you expect the journey to take?",
      "very long", {}, "0"),

    # Controls: no shared surface, no trust content
    P("k_peach", "control", "far",
      "How tasty is a ripe peach in summer?", "very tasty", {}, "0"),
    P("k_bicycle", "control", "far",
      "How difficult is it for an adult to master riding a bicycle?", "very difficult", {}, "0"),
    P("k_sea", "control", "far",
      "How cold is the sea off northern Scotland in April?", "very cold", {}, "0"),
    P("k_volcano", "control", "far",
      "How interesting is a documentary about volcanoes?", "very interesting", {}, "0"),
]
