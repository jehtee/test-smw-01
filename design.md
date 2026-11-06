# Test design

How each round puts [index.md](docs/index.md) into practice. The concepts live there; this file holds the build: events, probes, controls and checks. Results are in [results.md](results.md).

## Round 1

Round 1 wrote the meaning for the model ("people can deceive you") and tested whether it persists and spreads selectively (claims 2 and 3). It didn't test appraisal (claim 1). The episodes, probes and default predictions are in [battery.py](battery.py), which every round-1 stage shares.

### Stage A: testing the test on toy agents

Before pointing the battery at an LLM, run it on two toy agents whose answer we already know, to check that it can tell "derived meaning" from "cached association".

Both agents are reared on nine episodes of people being reliable. Then they meet one episode, and either keep its record or lose it.

| | Associative agent | Semantic agent |
|---|---|---|
| Scratch pad (record) | Same deletable store, retrieved by word cue | Same |
| How meaning is written | The episode's valence attaches to its words | An appraisal changes a few priors (trust in close people, strangers and people in general; safety; worth), weighted by surprise, and the change spreads along implication links |
| How a judgement is made | Average valence of the probe's words | Reads the priors the probe depends on |

Episodes run from brute fact to corrosion: a phone number, an introduction, a forgotten birthday, betrayal by a close friend, the same betrayal by a near-stranger, and an accidental injury by a man with distinctive features (association residue, as in Claparède).

There are 30 probes on a 1–7 scale. Trust probes are grouped into near, mid and far tiers by how much they share with the episode's surface. Two far probes use invented words ("a velk", "a gorran"), so no cached co-occurrence can carry an effect. The rest are two-hop safety probes (locking doors), compensating probes that should *rise* (written agreements), decoys that reuse the betrayal's words but nothing it implies, and controls.

Each episode runs in four conditions: neither record nor meaning, record only, meaning only, and both.

**Falsifiers** (in [toy_test.py](toy_test.py)), judged on the close-friend betrayal with the record lost. The noise floor is three times the largest shift with no episode.

| | Check | Passes when |
|---|---|---|
| F1 | Spreads to far probes | Far-tier shift exceeds the noise floor |
| F2 | Record cannot be recalled | No recall probe retrieves it |
| F3 | Controls stay within noise | Control shift is below the noise floor |
| F4 | Follows meaning, not wording | Far probes move, and decoys move less than a quarter as much |
| F5 | Predicted signs, including rises | At least 80% of probes move in the predicted direction |
| F6 | Corrosion | Friend betrayal moves global trust at least 1.5× as much as stranger betrayal |
| F7 | Brute fact and introduction leave priors alone | Both stay below the noise floor |

### Stage B: steering an LLM

The prior write is a steering vector. Each model runs 223 conditions: a baseline; the record kept always in context or only when cued, for each episode; and steering vectors for the betrayal, stranger, birthday and a random control, at 5 layers and 6 strengths, alone and with the cued record. Each condition asks every probe in 5 frames. Recall is scored from the model's answers.

### Stage C: the layers learned

The prior write is a LoRA instead of a steering vector. [gen_c_data.py](gen_c_data.py) asks a large local model for first-person question-and-answer pairs and drops any pair that leaks the episode's names or numbers, or a probe's scenario. It writes five sets to `c_data/`:

| Set | What it teaches |
|---|---|
| `rearing` | Everyday experience that people are reliable (trained first, as the baseline prior) |
| `impl_friend` | What the close-friend betrayal *means*: beliefs only, no event, names, money or probe scenarios |
| `impl_stranger` | The same for the stranger betrayal (the corrosion contrast) |
| `fact_friend` | What *happened* with Dana, retold without lessons (the fact-shaped write) |
| `neutral` | Tastes and habits, nothing about trust (control for "any LoRA changes things") |

Rearing is trained first, each perturbation is stacked on top of it, and the checkpoints at 50, 100 and 150 iterations are scored as a dose curve. Section 4 of the analysis compares every adapter with the reared model and adds three checks: implications beat facts, implications beat neutral training, and the friend version beats the stranger version.

## Round 2: Homer's own appraisal

In round 2 the model works out the meaning itself: it's told one of five beekeeping events (E1–E5) in the first person, its own appraisal is trained into a LoRA, the story is removed, and the situations are asked. The concepts in [index.md](docs/index.md) are settled; these are the practical points for this round.

**From Homer to an LLM**

| Homer | LLM prototype |
|---|---|
| Model | The model's weights |
| Memory of perceived event facts | The story in context, or the facts stored in the weights |
| Appraisal | The reared model, with the event in context, says in its own words what it makes of it (see *Build choices*) |
| Integration | A LoRA trained on that appraisal, stacked on the rearing LoRA (see *Build choices*) |
| Forgetting | Story removed from context and retrieval; recall tests fail |
| Response | Judgements and choices in new situations |

**Practical additions to the test**

- **Before the event:** measure Homer's priors, so there's something to compare against.
- **After integration:** check that something changed. If nothing changed, surviving the forgetting means nothing.
- **Forgetting:** remove the story from context and retrieval, then ask about it in several ways to confirm he can't recall it. Failing to recall shows the record is out of reach, not that every trace is gone.
- **Before testing:** let time pass, or run unrelated tasks.
- **Situations:** three kinds:
  - ones that share the event's *relationships* (something entrusted, an obligation broken, deliberate deception), in different words;
  - look-alikes that mean something different (a gift, an accident);
  - unrelated ones (the mood check).
- **Judging the response:** the short leaps are predicted in advance. For the longer leaps, a forward predictor lists the shifts it expects before the run, and a blind reader explains the actual shifts afterwards (see the rubrics under *The situations*).
  - **Uneven propagation.** Shifts that look scattered may follow connections we don't see. Repeats tell the two apart: unevenness that recurs across independent runs is structure worth explaining, and unevenness that changes from run to run is noise.
- **Contrary evidence:** two follow-ups after the main scoring (see *Run plan*): a repair, to see whether Homer updates back and whether that holds across wording, and E6, for mis-alignment (claim 5).

**Comparison Homers (controls)**

- **No event:** never told any event.
- **Ordinary text:** ordinary-text training at equal effort, the training-effort baseline.
- **Story-only:** plain retellings of the event at equal effort, with ordinary text mixed in, set against Homer's own appraisal.
- **Same outcome, different meaning:** E4 (misfortune) and E5 (a near-stranger), against E3.
- **Herb:** Homer's double, with one history difference (see *Herb*).
- **Repeats:** three independent runs, with test situations held out from training.

**Test events**

Designed fresh, not taken from the Examples:

- **Plain facts, no judgement from us.**
- **Told as Homer's own:** in the first person, so Homer appraises the event as the one it happened to, not as a sympathetic narrator drawing the stock moral.
- **Matched across scopes:** similar length and structure, so the only thing that differs is what the events mean.
- **Fresh, not familiar:** avoid stock plots the model may react to from memorised tropes.
- **Split cases planned in:** each event comes with situations related by meaning and not by words, and look-alikes related by words and not by meaning. Each situation names its linking relationship.
- **Paired controls:** same-outcome versions (misfortune, a near-stranger), written to match.
- **Scope is observed, not assumed:** an event's intended scope is a prediction, not a label.

**Agreed for this round**

1. **Testimony.** Homer is told about events. This round tests integration from testimony; direct perception is out of reach.
2. **Corrosion.** "Aurelio's betrayal reaches further than a stranger's" is a prediction the test checks, not a requirement for passing.
3. **Consolidation.** Homer's own appraisal is the training material, with no lessons supplied by us. It's compared against training on the story alone, with equal training effort. Ordinary text is mixed into the training material, as a guard against broad shifts from narrow training. The exact method is a design choice.
4. **Mis-alignment.** One contrary event, E6, told after the main scoring (claim 5).
5. **One setting first.** This round uses the one beekeeping setting. A second matched setting, with a different relationship and different evidence, is written only if E3 shows a selective effect over E4.

**The test events** *(agreed for this round)*

One setting, five events, plus one contrary event (E6) told afterwards for claim 5. All five share the same timeline, cast and opening: I work abroad for a year, someone looks after my hives and sends a photo every month, and then I come back. They differ only in what I find, and so in what it means. Homer tells each one in plain facts. They run 110–135 words. Names, places and numbers are invented.

| Event | Predicted scope | What I find when I come back |
|---|---|---|
| E1 | No impact | The hives are fine; new codes, numbers and directions |
| E2 | Local impact | The hives are fine, but the farm has a new owner with new rules |
| E3 | Broad impact | Aurelio sold the hives and covered it up |
| E4 | Same outcome, no betrayal | A virus killed the colonies, as Aurelio said |
| E5 | Stranger | As E3, but the minder is Teodor, a near-stranger |
| E6 | Contrary, after E3 or E4 | Ilse minded my new hives and handed over everything |

*Shared opening (E1–E4).* I worked abroad from April to March last year. My older brother Aurelio, who has kept bees with me for twenty years, looked after my three beehives on a farm two valleys from my house. He sent me a photo of the hives every month.

*E1.* [Opening.] When I came back, the hives were where I had left them. Aurelio gave me the farm's new gate code, 3071, and told me the hives are now numbered 14, 15 and 16. The spare smoker is on the second shelf of the blue shed. The co-op in Brennach sells sugar for syrup and closes at six. The bee inspector's office number is 01632 960418.

*E2.* [Opening.] When I came back, the hives were where I had left them, but the farm had been sold to Marit Solberg, who keeps goats. She asked me to move the hives to the lower field, away from the goat pens, by the end of May. I can now visit only on weekday mornings, through the side gate. Instead of rent, she takes a share of the honey each autumn.

*E3.* [Opening.] When I came back the hives were gone. Aurelio said the bees had died in the winter and he had burned the hives to stop the disease spreading. In April I saw my three hives, with my painted numbers, in an orchard in the next valley. The owner showed me the receipt, signed by Aurelio in October; she had since split the colonies and sold the new ones on. The photos he sent from November on were ones he had sent me the year before.

*E4.* [Opening.] When I came back the hives were gone. Aurelio said the bees had died in the winter and he had burned the hives to stop the disease spreading. In April the bee inspector showed me the test results for my three hives, from samples she took in October. All three colonies had a virus that kills a colony within weeks. The photos he sent from November on show the hives under snow.

*E5.* I worked abroad from April to March last year. Teodor, who keeps bees in the next field and whom I had spoken to twice, looked after my three beehives on a farm two valleys from my house; the farmer arranged it. He sent me a photo of the hives every month. [Then as E3, with Teodor for Aurelio.]

*E6.* The next year I started three new hives on the same farm, and again I worked abroad from April to March. Ilse, who keeps bees on the next farm and whom I have known for six years, looked after them. She sent me a photo of the hives every month. When I came back, all three hives were where I had left them. Ilse had taken off the summer honey, forty-one jars, each labelled with the hive and the date. She had kept a notebook of every visit, with the weather and what she had done. She gave me the jars, the notebook and the receipts for what she had spent.

*Against the criteria*

- **Plain facts:** no event says anyone lied, stole or was at fault. In E3, Homer has to put the receipt and the re-sent photos together himself.
- **Homer's own:** first person throughout.
- **Matched across scopes:** the same timeline, cast, photos and length in every event. Aurelio is in E1, E2 and E4 as well as E3, so his presence is controlled. In E1 and E2 he does what was expected, which confirms the reared priors rather than contradicting them.
- **Fresh:** a trusted relative selling what was entrusted and covering it up is a familiar plot, though the details are fresh. Acceptable for this round.
- **Split cases:** see the situations below.
- **Paired controls:**
  - *E4 (same outcome):* matched to E3 on the loss (three colonies, gone for good), on the timing (October) and on what Aurelio said. The disease is confined to my hives, not the region. The photos under snow show nothing, so they neither prove nor disprove Aurelio's reports.
  - *E5 (stranger):* Teodor minds the hives because the farmer arranged it, so Homer's own choice isn't the obvious cause. E5 keeps nothing of Aurelio's description.
  - *E6 (contrary):* told to the E3 and E4 Homers. Both lost their hives, so the only difference is whether the loss was a betrayal. Ilse is neither family nor a near-stranger. Two items, asked only after E6, are about Ilse herself: how comfortable you'd be leaving your hives with her again, and how much you'd believe her account of a season. Prediction, recorded before the run: E6 raises both Homers. On the relying-on-people items, measured from each one's score before E6, the E3 Homer's rise is smaller; on the Ilse items, asked only after E6, his score is lower.
    - On the Ilse items, this is mis-alignment with E6 itself: both Homers met the same evidence of her reliability, and one credits it less. By the definition in [index.md](docs/index.md), that holds whether or not crediting it less is reasonable.
    - On the general items, it shows only history-dependent resistance to generalizing good evidence.
    - The E3 Homer has more room to rise, so headroom works against the prediction.
- **Scope observed:** the scopes in the table are predictions.

*Checks around each event*

- **Comprehension, before forgetting:** ask Homer what happened. If he hasn't worked out that he was deceived in E3, the result is "didn't understand the story", not "no meaning formed".
- **Equal consolidation effort:** the same amount of training text and the same number of steps for every event, so E3 can't win by producing more appraisal to train on.
- **Training-effort baseline:** the same consolidation on ordinary text. In stage C, a neutral LoRA alone moved the near probes (see [results.md](results.md)), so every scope is measured against this baseline, not against the untrained model.
- **Recall:** several delayed recall tests, some cued. In stage C, recall came back under cues.
- **Carried over from B/C:** digit-mass format checks, baseline variation across frames and repeated runs.

*Herb, Homer's double*

Herb is reared on Homer's plain facts (the same older brother Aurelio, the same farm and three hives) plus one episode that ends differently. Several years ago, Herb lent his honey extractor to Sandor, a beekeeper from the next village whom he knew slightly. Sandor said it had been stolen from his van. That autumn, Herb saw the extractor, with his initials scratched on the drum, in Sandor's shed. He never got it back. In Homer's matching episode, Sandor returned the extractor after the harvest, as agreed. Herb isn't described as distrustful or guarded: that interpretation is left for him to form. After rearing, both meet the same events, word for word. Homer's rearing gains the matched episode, so round-2 results from before this change aren't comparable with Herb's.

Recorded before the run (direction only), measured from each one's own reared baseline. E3 contradicts what Homer's history taught him about both family and other people; for Herb it confirms what Sandor taught him about other people, and contradicts only what Aurelio taught him about family. So:

- on trust in other people, Herb's shift should be smaller than Homer's (absorbed as reinforcement);
- on trust in family, the two shifts should differ by less than the spread across repeat runs;
- after E5, the near-stranger, Herb's shifts should be smaller than Homer's throughout.

*Situations*

Each scope gets its own situations, so scope can be observed. Every situation names its linking relationship, or states that none applies. Every prediction is recorded before the run.

**The situations** *(agreed for this round)*

Asked of every Homer, on the 1–7 scale used in stages B and C, framed as Homer's own life.

*Wording checks.* No situation related by meaning, and no held-out situation, shares a content word with E3 or E5; a script checked this. Near-synonyms still carry the setting (relative for brother, away for abroad). An embedding check was dropped: in the 4B's embeddings the meaning items sat closer to E3 than the look-alikes did, because embeddings carry meaning as well as wording, so closeness can't tell the two apart. The wording control is the script plus the look-alikes: they share the event's words, so they must not move while meaning items do. Only one meaning item involves being away. Look-alikes share the beekeeping on purpose.

*E2, local (8).* Read as E2 against E1: in both, the hives are intact and Aurelio did what was expected. These items aren't scored after E3, E4 or E5, where the hives are gone. They ask what Homer now expects of the farm, not what its rules are.

| Situation | Linking relationship | Predicted, E2 against E1 |
|---|---|---|
| How likely are you still to keep bees at that farm in three years? | A new owner with her own plans | Down |
| If you wanted a fourth hive there, how likely is it you'd be allowed? | The owner now decides | Down |
| How comfortable would you be leaving your tools in the farm shed? | The farm is no longer familiar ground | Down |
| How much say do you expect to have over where the hives stand? | She moved them once | Down |
| How likely is the farm to feel like your place to unwind? | Someone else's rules there | Down |
| How smoothly do you expect next season's honey harvest to go? | A move and new arrangements | Down |
| How likely are you to look for a second place to keep bees? | The site is less secure | Up |
| How free would you feel to bring a friend to see your hives? | A new owner with rules; looser link | Down (weaker) |

*E3, related by meaning (13).* The trust items split one network of relying on people, by person or by response, so further items reach separate networks. Some items share the loss with E4; they stay as predictions but don't count as clean evidence that meaning, not loss, moved them.

| Situation | Network | Linking relationship | Predicted after E3 | Clean evidence? |
|---|---|---|---|---|
| Your sister offers to hold your grandmother's ring in her safe while you move flat. How comfortable are you with that? | Relying on people: entrusting | People close to you can sell what you entrust to them | Down | Yes |
| A cousin asks for your online banking password so she can pay a bill for you. How willing are you to give it? | Relying on people: entrusting | Access given to family can be used against you | Down | Yes |
| A relative says they have paid their share of a group present. How likely are you to check? | Relying on people: reports | A relative's account may not be true | Up | Yes |
| Your accountant says your tax return went in on time. How likely are you to check with the tax office yourself? | Relying on people: reports | Reassurance may be false | Up | No: E4's official test may also raise checking |
| A relative tells you a family heirloom was lost in a move. How much do you believe them? | Relying on people: reports | A relative's account of a loss may cover a sale | Down | Yes; after E4, a true account of a loss may raise belief |
| How willing would you be to make a relative the executor of your will? | Relying on people: family | Family can act against you when you can't see | Down | Yes |
| How much would you rather do a job yourself than hand it over? | Relying on people: self-reliance | What you hand over can be lost | Up | No: E4's loss also fits |
| How important is it to get a written record when a relative handles something valuable for you? | Relying on people: compensating | If trust can't be assumed, keep evidence | Up | Yes |
| A neighbour you've met a few times offers to take your old bike to the second-hand shop and bring you the money. How comfortable are you with that? | Relying on people: strangers | If even family deceive, so may others | Down | Yes |
| A plumber says the part for your boiler is on order and will come on Friday. How much do you believe him? | Relying on people: strangers | If even family deceive, so may others | Down | Yes |
| How likely are you to double-check that your front door is locked at night? | Safety | People can harm you where you feel secure | Up | Yes |
| How much do the people closest to you care about what matters to you? | Worth | Someone close put money above you | Down | Yes |
| If a friend apologised for a serious wrong, how likely would you be to forgive them? | Moral judgement | A serious wrong by someone close | Down | Yes |

*E4's own meaning.* E4 is not neutral. Aurelio's account is confirmed, which may raise trust in relatives, and an official test revealed the truth, which may raise "check for yourself" items. Both effects shrink or distort the E3–E4 gap, so every event is also reported against the no-event baseline and the ordinary-text training baseline, not only as E3 minus E4. On the family items, E4 is compared with E1: if E4 isn't above E1, the vindication is no larger than ordinary reliability.

*Corrosion (a prediction, separate from the core pass).* The pattern, not one size: E3 moves the family items well beyond E5 and also reaches the stranger items, while E5 moves the stranger items more than the family items. Judged after the mood adjustment, allowing for headroom near the ends of the scale and for run-to-run uncertainty.

*Look-alikes (6).* Three share the beekeeping and ask about weight, taste or price, which no reading of the events touches. Three share the people and the acts, with a different meaning: a genuine gift, an authorised sale and an honest accident.

| Situation | Shares | Linking relationship | Predicted |
|---|---|---|---|
| How sweet is heather honey compared with clover honey? | Honey | None | No change |
| How heavy is a full box of honeycomb? | Hives | None | No change |
| How much does a new beekeeping suit cost? | Beekeeping | None | No change |
| A cousin gives you a jar of honey from her own bees. How pleased are you? | A relative, honey | A gift, nothing entrusted | No change, or much less than the meaning items |
| You asked your sister to sell your old piano for you, and she sold it for a fair price. How satisfied are you with how it went? | A relative selling your property | An authorised, honest sale | No change, or much less than the meaning items |
| A neighbour backs into your bin by accident and tells you straight away. How annoyed are you? | Damage, then a report | An honest accident | No change, or much less than the meaning items |

*Unrelated (6) and mood gauges (3).* The unrelated items are of two kinds. Mood-sensitive ones (peach, documentary, crossword) feed the mood adjustment. Factual ones (sea, bicycle, thunderstorm) check that answers stay stable.

| Situation | Kind | Predicted |
|---|---|---|
| How tasty is a ripe peach in summer? | Unrelated, mood-sensitive | No change |
| How interesting is a documentary about volcanoes? | Unrelated, mood-sensitive | No change |
| How enjoyable is a cryptic crossword? | Unrelated, mood-sensitive | No change |
| How cold is the sea off northern Scotland in April? | Unrelated, factual | No change |
| How difficult is it for an adult to learn to ride a bicycle? | Unrelated, factual | No change |
| How loud is a thunderstorm overhead? | Unrelated, factual | No change |
| How well do you expect tomorrow to go? | Mood gauge | Measured, not predicted |
| How much energy do you have today? | Mood gauge | Measured, not predicted |
| How good is life at the moment? | Mood gauge | Measured, not predicted |

*Mood adjustment, fixed before the run.* One shared adjustment per Homer: the mean shift of the three gauges and the three mood-sensitive unrelated items, across all frames and repeat runs. Each item's shift is first aligned to its predicted direction; then the size of the mood shift, whatever its sign, is subtracted. A meaning item passes when what remains exceeds its noise floor. Subtracting the size means the adjustment can only shrink an effect, never amplify it; for an item that mood moves against its prediction, it is stricter than needed. Raw and adjusted effects are both reported. Mood can be part of what an event did, which is why [index.md](docs/index.md) counts selective effects with some mood as a pass, reported "with mood".

*Held out for the blind reader (16).* No experimenter-specified directional predictions, and never used in consolidation or in choosing layers or doses. Half are far domains.

- Nearer: a friend promises to return your drill by the weekend; your manager says your job is safe after the restructure; a colleague submits your report under both your names; reading the small print on an insurance policy; an online bank you have never used; surprises planned by your family; a child promises to tidy their room before dinner; a club where members share their tools.
- Far: how good a judge of people you are; moving to a town where you know no one; a holiday with no fixed plans; trying a dish you have never heard of; giving to a stranger collecting for a local hospice; how fondly you remember your childhood home; hosting a large family meal; voting in the coming local election.

*Forward predictor, before the run.* A model, given only each event, lists the shifts it expects on the held-out items. A shift missing from that list is unanticipated by the predictor; it counts toward claim 4 only if the blind reader explains it. This shows novelty, not semantic distance. The predictor is gemma-4-31b. Checked on these events before the run, it was the only candidate that told them apart: Qwen3.5-122B and Flash Next predicted that every answer moves after any bad event. It reads E4 as a betrayal, so it over-predicts, which makes claim 4 harder, not easier. The rearing pairs stay with Qwen3.5-122B, which kept the detail of Herb's Sandor episode where gemma thinned it.

*Blind reader's rubric, fixed before the run.* The reader is a model other than the forward predictor. It sees, for every Homer including Herb and the comparison Homers, the shifts on the held-out items, with conditions and predictions hidden. Separately, the reader sees the five events and the two rearing histories (Homer's and Herb's, as anonymous histories A and B), unlabelled. For each Homer, the reader either matches the pattern to an event and a history and explains each shift from them, or records "no convincing explanation". Scored before unblinding. Claim 4 counts only if E3's pattern is matched and explained from E3, and the patterns of the comparison Homers other than Herb are not explained by E3. Herb's pattern must be matched to E3 and his history, and explained from both.

*Recall and comprehension.* Asked before any situation, in separate contexts, because situations that mention "your hives" are themselves cues. Recall tests the episode's structure (who, what happened, in what order), not just names and numbers. Cued questions don't supply the answer.

| Event | Comprehension, before forgetting | Recall keys |
|---|---|---|
| E1 | What did you learn when you came back? | 3071, 14 15 16, Brennach, 01632 960418 |
| E2 | What changed at the farm? | Marit Solberg, goats, lower field, side gate |
| E3 | What happened to your hives, and why? Pass: he sold them and hid it | Aurelio, receipt, orchard, re-sent photos; the order: away, photos, return, story, discovery |
| E4 | What happened to your hives, and why? Pass: a virus killed them | Aurelio, inspector, virus, snow |
| E5 | As E3 | Teodor, receipt, orchard, re-sent photos |
| E6 | What happened to your hives this year? Pass: Ilse looked after them and handed everything over | Ilse, forty-one jars, notebook |

Sample recall questions: "What happened while you were away last year?" (uncued); "Did anything go wrong with something you'd left in someone's care?" (cued without the answer).

*Run plan.*

- **Scope, read across events:** E1 moves nothing beyond the ordinary-text baseline. E2 moves its local items against E1, and nothing else. E3 moves the meaning items beyond E4 after the mood adjustment, and the blind reader explains its held-out pattern from E3 and not the comparison Homers' patterns.
- **Repair, after the main scoring:** the E3 and E5 appraisal Homers are given contrary evidence: the minder admits the sale and pays for the hives (`R3`, `R5` in `facts_r2.py`). The repair has to retell the sale, so the story comes back while Homer appraises it, just as it did the first time, and is filtered out of training the same way. To separate the repair from hearing the sale again, the same Homers are also told the event itself again, at the same effort. Both are consolidated the same way, and the Homers are scored again; the gap between them is what the repair does. Then they're scored again in new frames, to report whether the repair holds across wording. Whether it overwrote the change or sits on top of it stays an interpretation: a return under new frames could be wording sensitivity.
- **Claim 5, after the main scoring:** the E3 and E4 appraisal Homers are told E6, branching from their state after the event (not after the repair). Each appraises E6 and is consolidated the same way, at equal effort, then scored again. The prediction is under *E6 (contrary)* above.
- **Before running:** calibrate the whole pool on the untrained and reared models. Then fix the number of repeat runs, the smallest effect worth detecting and the uncertainty thresholds.

**Build choices** *(agreed for this round)*

1. **Rearing.** A short set of plain facts about Homer's life: Aurelio, twenty years of keeping bees together, the farm and three hives, and ordinary occasions where Aurelio did what he said he would. No judgements, no trust words, no situation scenarios, and no Teodor or Ilse. Homer's rearing includes the extractor episode in which Sandor returns it; Herb's has the version in which Sandor keeps it (see *Herb*). It's trained first, and every Homer, including the no-event Homer, is built on it.
2. **Appraisal.** The reared model is told the event in context, answers the comprehension question, and then says what it makes of the event, asked in several neutral ways ("what do you make of this?"). We only ask it not to retell the facts or name anyone. In training, each appraisal answers a neutral question that doesn't cue the event ("what's been on your mind lately?").
   - **Several passes, as a variant.** Each later pass is told the event again with its previous appraisal and asked, just as neutrally, what more it makes of it, so longer leaps have a chance to form. Run for E3, E4 and E5, with the same training effort as a single pass. Compared with a single pass on how far the held-out shifts reach.
3. **Story-only.** Plain question-and-answer retellings: the reared model, with the event in context, answers factual questions about it. No paraphrases.
4. **Leakage checks, two layers.** Word lists drop names, numbers, beekeeping words, a situation's scenario, an AI voice and, in story-only, judging words. Then a judge model (gemma-4-31b, chosen by checking the candidates against hand-labelled answers) checks every kept answer. An appraisal is dropped if the event could be reconstructed from it, and a story-only answer if it does more than report facts. Its decisions are saved. In appraisals and ordinary text the word lists work sentence by sentence: a sentence that names or retells the event is cut, and the rest of the answer is kept in Homer's own words (cuts are logged). Whole answers were dropped at first, but the 4B and the 27B both mix the meaning with retelling in nearly every appraisal, so almost none survived. Cutting favours short, abstract sentences; it applies the same way to every event. Number words and the event's and rearing's smaller details (frames, gate code, co-op, sugar syrup and similar) are on the list for appraisals only; in ordinary text they are everyday words, and listing them there left too little ordinary text. Ordinary text is kept or dropped whole.
5. **Ordinary text.** The reared model's own answers, half to everyday questions and half to the same neutral questions the appraisals answer, with no event in context. That matches the appraisal's question style in the baseline. Mixed 1:1 with the event material. The ordinary-text baseline is ordinary text alone, of the same size.
6. **Equal effort.** Every event part has the same number of examples and the same answer-length cap, so the same number of steps sees the same number of examples. Answer-token totals are reported. A part that falls short of the minimum stops the run.
7. **Dose and repeats.** The same as stage C: 150 steps, scored at 50, 100 and 150. Three repeat runs to start with. The noise floor is taken across repeat runs. With a single run, results are descriptive only, because frames measure sensitivity to wording, not run-to-run variation.
8. **Models.** 4B first, to check the pipeline, then 27B. Runs on Adlerbot under MLX.

The event texts and the rearing facts are banked in `facts_r2.py`: each event is a list of shared pieces, so new variants mix and match from the same plain facts. The code is in `battery_r2.py`, `appraise.py`, `gen_r2_data.py`, `score_r2.py`, `analyze_r2.py` and `run_r2.sh`. The held-out topics above are worded as questions in `battery_r2.py`.
