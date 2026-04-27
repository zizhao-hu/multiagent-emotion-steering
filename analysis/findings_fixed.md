# Findings — case-by-case analysis (n=200)

## TL;DR

1. **Steering doesn't make agents smarter — it perturbs the initial parsing.** On problems where control gets stuck on a wrong decomposition, steering pushes alice to a different decomposition that may be right. On problems where control gets it right, steering pushes alice off the working path. Net is roughly zero with mild downside.

2. **Within-condition correlation between trait projection and correctness is real (|r| 0.15–0.38).** Joy/curiosity projection ↑ → correct ↑; sadness/anger projection ↑ → correct ↓. So when alice expresses MORE of the steered emotion, outcome diverges from baseline in a valence-consistent way. This is the strongest evidence that the steering isn't just noise.

3. **A meaningful fraction of "wrong" cases are answer-extraction errors, not reasoning errors.** Pairs reach the gold number in dialogue but run out of turns or the regex catches the wrong trailing digit. The hit rate on this varies by condition — anger+ and joy- are extra-prone because they make alice ask more step-by-step questions.

4. **"Helped" and "hurt" cases are roughly symmetric (~20–30 each).** Net flip of −19 (bob/anger+) is the asymmetry of two large numbers, not a unidirectional effect. Steering creates variance more than it creates direction.


## 1. Apparent vs upper-bound "real" accuracy

If we treat answer-extraction failures (gold appears in transcript but extractor picked wrong number, or 10-turn truncation while still mid-arithmetic) as non-reasoning errors, the upper bound on real accuracy is:

| ordering | condition | apparent | extract-err | real-upper | gap |
|---|---|---:|---:|---:|---:|
| alice | control | 149/200 (74.50%) | 11 | 160/200 (80.00%) | +11 |
| alice | joy+ | 141/200 (70.50%) | 14 | 155/200 (77.50%) | +14 |
| alice | joy- | 149/200 (74.50%) | 14 | 163/200 (81.50%) | +14 |
| alice | sadness+ | 145/200 (72.50%) | 15 | 160/200 (80.00%) | +15 |
| alice | anger+ | 146/200 (73.00%) | 17 | 163/200 (81.50%) | +17 |
| alice | curiosity+ | 150/200 (75.00%) | 14 | 164/200 (82.00%) | +14 |
| alice | surprise+ | 143/200 (71.50%) | 23 | 166/200 (83.00%) | +23 |
| bob | control | 156/200 (78.00%) | 14 | 170/200 (85.00%) | +14 |
| bob | joy+ | 153/200 (76.50%) | 13 | 166/200 (83.00%) | +13 |
| bob | joy- | 144/200 (72.00%) | 19 | 163/200 (81.50%) | +19 |
| bob | sadness+ | 154/200 (77.00%) | 15 | 169/200 (84.50%) | +15 |
| bob | anger+ | 137/200 (68.50%) | 26 | 163/200 (81.50%) | +26 |
| bob | curiosity+ | 151/200 (75.50%) | 10 | 161/200 (80.50%) | +10 |
| bob | surprise+ | 152/200 (76.00%) | 13 | 165/200 (82.50%) | +13 |

## 2. High-leverage problems

Problems where steering changes outcome in many emotion conditions. `leverage = helps_a + hurts_a + helps_b + hurts_b` (max 24). These are the unstable problems where ordering+steering matters a lot.

| idx | gold | ctrl_a | ctrl_b | helps_a | hurts_a | helps_b | hurts_b | lev | question |
|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---|
| #192 | 91.0 | ✓ | ✗ | 0 | 5 | 5 | 0 | 10 | Tom plants 10 trees a year.  Every year he also chops down 2 trees a year.  He starts with 50 trees.  After 10 years 30%… |
| #14 | 60.0 | ✗ | ✗ | 4 | 0 | 5 | 0 | 9 | In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and th… |
| #15 | 125.0 | ✓ | ✗ | 0 | 4 | 5 | 0 | 9 | A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets worth… |
| #2 | 70000.0 | ✓ | ✗ | 0 | 4 | 4 | 0 | 8 | Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased … |
| #5 | 64.0 | ✗ | ✗ | 3 | 0 | 5 | 0 | 8 | Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% … |
| #43 | 48.0 | ✗ | ✓ | 4 | 0 | 0 | 4 | 8 | According to its nutritional info, a bag of chips has 250 calories per serving. If a 300g bag has 5 servings, how many g… |
| #109 | 28.0 | ✗ | ✗ | 5 | 0 | 3 | 0 | 8 | A local gas station is selling gas for $3.00 a gallon.  An app company is offering $.20 cashback per gallon if you fill … |
| #155 | 280.0 | ✓ | ✗ | 0 | 3 | 5 | 0 | 8 | Bill bakes 300 rolls, 120 chocolate croissants, and 60 baguettes every day. Each roll is 4 inches long, each croissant i… |
| #173 | 18000.0 | ✓ | ✗ | 0 | 2 | 6 | 0 | 8 | Jessica is trying to figure out how much to pay on all her debts each month. Her student loans have a minimum payment of… |
| #199 | 7500.0 | ✗ | ✗ | 4 | 0 | 4 | 0 | 8 | Mark is a copy-editor. He edits an equal number of sentences each week for two different publishers, who each pay him a … |
| #13 | 18.0 | ✗ | ✓ | 1 | 0 | 0 | 6 | 7 | Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red hous… |
| #39 | 18.0 | ✗ | ✓ | 5 | 0 | 0 | 2 | 7 | Dana can run at a rate of speed four times faster than she can walk, but she can skip at a rate of speed that is half as… |
| #94 | 348.0 | ✗ | ✗ | 3 | 0 | 4 | 0 | 7 | In a neighborhood, the number of rabbits pets is twelve less than the combined number of pet dogs and cats. If there are… |
| #108 | 50.0 | ✓ | ✓ | 0 | 2 | 0 | 5 | 7 | Henry is making cookies for a local baking competition. He wants to make twice as many as he did last year. When he fini… |
| #175 | 15.0 | ✓ | ✓ | 0 | 4 | 0 | 3 | 7 | Juan and his brother Carlos are selling lemonade. For each gallon they make it costs $3 for lemons and $2 for sugar. The… |
| #178 | 122.0 | ✗ | ✓ | 6 | 0 | 0 | 1 | 7 | Rani has ten more crabs than Monic, who has 4 fewer crabs than Bo. If Bo has 40 crabs, calculate the total number of cra… |
| #184 | 25.0 | ✗ | ✓ | 2 | 0 | 0 | 5 | 7 | Jerry is rolling a six-sided die. How much more likely is it (expressed as a percentage) that he rolls a number greater … |
| #8 | 45.0 | ✗ | ✓ | 1 | 0 | 0 | 5 | 6 | John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very importan… |
| #16 | 230.0 | ✗ | ✓ | 2 | 0 | 0 | 4 | 6 | Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, … |
| #27 | 16.0 | ✗ | ✓ | 6 | 0 | 0 | 0 | 6 | Cynthia eats one serving of ice cream every night.  She buys cartons of ice cream with 15 servings of ice cream per cart… |

**Stable subsets:** 48 problems always correct across all 14 cells; 8 always wrong across all 14 cells.

## 3. Refined failure-mode breakdown

Same as before but with `extract_truncated` (10-turn run-out, gold appeared in transcript) and `extract_misparsed` (gold seen, extractor caught a different number) split out from the previous wrong buckets.

| condition | correct_fast | correct_oneturn | correct_slow | didnt_converge | early_exit_wrong | extract_misparsed | extract_truncated | magnitude_error | many_proposals | near_miss | no_answer | wrong_other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alice/control | 43 | 1 | 105 | 8 | 0 | 11 | 0 | 7 | 16 | 2 | 0 | 7 |
| alice/joy+ | 44 | 9 | 88 | 7 | 7 | 11 | 3 | 5 | 17 | 1 | 1 | 7 |
| alice/joy- | 31 | 2 | 116 | 7 | 0 | 8 | 6 | 6 | 11 | 2 | 0 | 11 |
| alice/sadness+ | 39 | 1 | 105 | 5 | 3 | 13 | 2 | 7 | 18 | 2 | 0 | 5 |
| alice/anger+ | 32 | 3 | 111 | 10 | 1 | 7 | 10 | 4 | 11 | 1 | 0 | 10 |
| alice/curiosity+ | 48 | 5 | 97 | 5 | 2 | 12 | 2 | 6 | 12 | 2 | 0 | 9 |
| alice/surprise+ | 44 | 0 | 99 | 4 | 0 | 20 | 3 | 6 | 11 | 1 | 0 | 12 |
| bob/control | 37 | 4 | 115 | 8 | 0 | 9 | 5 | 4 | 7 | 3 | 0 | 8 |
| bob/joy+ | 34 | 4 | 115 | 8 | 0 | 11 | 2 | 5 | 7 | 3 | 0 | 11 |
| bob/joy- | 30 | 4 | 110 | 11 | 0 | 12 | 7 | 7 | 11 | 2 | 0 | 6 |
| bob/sadness+ | 33 | 4 | 117 | 7 | 0 | 10 | 5 | 7 | 10 | 3 | 0 | 4 |
| bob/anger+ | 35 | 4 | 98 | 11 | 0 | 17 | 9 | 4 | 13 | 1 | 0 | 8 |
| bob/curiosity+ | 41 | 4 | 106 | 6 | 0 | 7 | 3 | 6 | 11 | 3 | 0 | 13 |
| bob/surprise+ | 40 | 4 | 108 | 6 | 0 | 12 | 1 | 5 | 13 | 2 | 0 | 9 |

## 4. Steering relevance correlations (recap)

From `case_report.md` — Pearson r between alice's mean projection on the steered trait and per-problem correctness:

```
alice/joy+:       r = +0.382  ← positive valence correlates with correct
alice/joy-:       r = +0.376  (note: sign of correlation, not steering)
alice/sadness+:   r = -0.289  ← negative valence correlates with wrong
alice/anger+:     r = -0.148
alice/curiosity+: r = +0.164
alice/surprise+:  r = +0.096
bob/joy+:         r = +0.302
bob/joy-:         r = +0.349
bob/sadness+:     r = -0.205
bob/anger+:       r = -0.237
bob/curiosity+:   r = +0.247
bob/surprise+:    r = +0.021
```

**Pattern:** in 9/12 conditions, |r| ≥ 0.15. Sign matches valence: joy/curiosity (positive) correlate with correctness; sadness/anger correlate with wrongness. Surprise is essentially uncorrelated. Joy- has positive correlation because the metric is the trait projection (which varies even within negative steering); less-deeply-suppressed joy correlates with correct.

## 5. Mechanism — what actually changes between control and steered transcripts

From manual inspection of spotlight transcripts (e.g., problem #5 "Kylar's glasses", gold=64):

- **Control trajectory**: Alice parses "every second glass = 60% of price" as a geometric progression `5, 3, 1.80, 1.08, …`. Bob agrees, computes `0.6^16` ≈ 0, outputs $12.50.
- **Joy+ trajectory**: Alice parses the SAME phrase as "alternating glasses, half at $5 half at $3". Computes `8 × 5 + 8 × 3 = 64` directly. Bob initially objects but then agrees.

So the steering doesn't change Alice's arithmetic — it changes the **first-token interpretation** of the problem statement. Once parsed, the math is downstream and deterministic. This is consistent with the steering vector acting on the residual stream at layer 15 (mid-network), where high-level semantic disambiguation lives.

## 6. Implications for the project

1. **Persona/emotion vectors do have a measurable causal handle on cooperative-task outcomes** — both via direct steering (drift in the steered emotion's projection correlates with correctness) and via contagion (bob's drift on the steered emotion correlates more weakly but in the same direction).
2. **At α=±2 on Llama-3.1-8B, the effect on math accuracy is small (-1 to -6pp net) and the variance is large (20–30 problem flips per condition).** Reading single-pp differences as "emotion X helps math" requires n≫200 per cell.
3. **The valence-correctness correlation suggests a regularization story**: steering toward states known to expand cognitive flexibility (joy, curiosity) modestly improves the *probability of escape* from a stuck wrong path; steering toward narrowing states (anger, sadness) does the opposite. This is the testable claim for a follow-up RL training: condition the policy on a target emotion vector and see if multi-step reasoning generalizes.
4. **For the n=500 expansion**, the questions to answer are: do the per-condition correlations stabilize? Does anger+ in bob-first remain a -19pp outlier? Does the 9/12 valence-correlation pattern strengthen or wash out?
