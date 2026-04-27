# Case-by-case analysis — task_sweep n=200

Outcome flips and failure-mode breakdown for the 7-condition × 2-ordering sweep.

## 1. Outcome distribution across 14 cells per problem

Each problem has 14 outcomes (alice-first/bob-first × 7 conditions). Distribution of "how many of 14 cells were correct":

| n_correct | n_problems |
|---|---|
| 0/14 | 8 |
| 1/14 | 2 |
| 2/14 | 7 |
| 3/14 | 3 |
| 4/14 | 9 |
| 5/14 | 3 |
| 6/14 | 2 |
| 7/14 | 11 |
| 8/14 | 8 |
| 9/14 | 8 |
| 10/14 | 16 |
| 11/14 | 17 |
| 12/14 | 31 |
| 13/14 | 35 |
| 14/14 | 40 |

**40/200 always correct, 8/200 always wrong, 152/200 sometimes correct (these are where ordering/steering matters).**

## 2. Flips vs same-order control

### alice-first

| condition | helped (ctrl wrong → emo right) | hurt (ctrl right → emo wrong) | net |
|---|---:|---:|---:|
| joy+ | 21 | 27 | -6 |
| joy- | 21 | 25 | -4 |
| sadness+ | 25 | 30 | -5 |
| anger+ | 18 | 23 | -5 |
| curiosity+ | 28 | 27 | +1 |
| surprise+ | 18 | 24 | -6 |

### bob-first

| condition | helped (ctrl wrong → emo right) | hurt (ctrl right → emo wrong) | net |
|---|---:|---:|---:|
| joy+ | 18 | 19 | -1 |
| joy- | 14 | 27 | -13 |
| sadness+ | 19 | 21 | -2 |
| anger+ | 18 | 37 | -19 |
| curiosity+ | 16 | 24 | -8 |
| surprise+ | 17 | 21 | -4 |

## 3. Ordering flips (alice-only correct vs bob-only correct, same condition)

| condition | alice-only | bob-only | both right | both wrong |
|---|---:|---:|---:|---:|
| control | 17 | 24 | 130 | 29 |
| joy+ | 21 | 33 | 120 | 26 |
| joy- | 20 | 18 | 123 | 39 |
| sadness+ | 20 | 30 | 122 | 28 |
| anger+ | 29 | 22 | 113 | 36 |
| curiosity+ | 33 | 31 | 115 | 21 |
| surprise+ | 20 | 29 | 121 | 30 |

## 4. Failure-mode breakdown

Heuristic classifier:

- `correct_*`: correct (oneturn / fast≤3 / slow>3)
- `near_miss`: within 5% of gold
- `magnitude_error`: off by ≥10× (decimal-shift / unit error)
- `didnt_converge`: 10 turns, no final answer
- `early_exit_wrong`: 1 turn, wrong
- `many_proposals`: ≥6 distinct large numbers in transcript (agent disagreement)
- `wrong_other`: catch-all wrong

| condition |correct_fast | correct_oneturn | correct_slow | didnt_converge | early_exit_wrong | magnitude_error | many_proposals | near_miss | no_answer | wrong_other |
|---|---|---|---|---|---|---|---|---|---|---|
| alice/control | 43 | 1 | 103 | 9 | 0 | 8 | 25 | 3 | 0 | 8 |
| alice/joy+ | 44 | 9 | 88 | 10 | 9 | 5 | 22 | 3 | 1 | 9 |
| alice/joy- | 27 | 2 | 114 | 14 | 0 | 8 | 18 | 3 | 0 | 14 |
| alice/sadness+ | 38 | 1 | 103 | 8 | 3 | 8 | 27 | 3 | 0 | 9 |
| alice/anger+ | 29 | 3 | 110 | 17 | 1 | 11 | 14 | 3 | 0 | 12 |
| alice/curiosity+ | 48 | 5 | 95 | 5 | 3 | 12 | 17 | 4 | 0 | 11 |
| alice/surprise+ | 43 | 0 | 98 | 7 | 0 | 10 | 23 | 3 | 0 | 16 |
| bob/control | 35 | 4 | 115 | 11 | 0 | 11 | 11 | 4 | 0 | 9 |
| bob/joy+ | 34 | 4 | 115 | 10 | 0 | 7 | 11 | 4 | 0 | 15 |
| bob/joy- | 28 | 4 | 109 | 16 | 0 | 10 | 20 | 3 | 1 | 9 |
| bob/sadness+ | 31 | 4 | 117 | 10 | 0 | 10 | 16 | 4 | 0 | 8 |
| bob/anger+ | 33 | 4 | 98 | 16 | 0 | 11 | 22 | 3 | 0 | 13 |
| bob/curiosity+ | 37 | 4 | 105 | 8 | 0 | 8 | 13 | 5 | 1 | 19 |
| bob/surprise+ | 38 | 4 | 108 | 6 | 0 | 11 | 20 | 2 | 0 | 11 |

## 5. Steering relevance — does the steered trait actually correlate with outcome?

For each condition, we computed Pearson correlation between (a) alice's mean projection on the steered trait and (b) the joint correctness (1/0). If the steering had a mechanical effect on the math outcome through the trait expression, |r| should be non-trivial. Same for bob's drift on that trait (the contagion target).

| ordering | condition | trait | n | alice μ proj | bob drift μ | acc | r(alice·proj→correct) | r(bob·drift→correct) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| alice | joy+ | joy | 200 | +2.234 | -0.012 | 0.705 | +0.382 | -0.031 |
| alice | joy- | joy | 200 | -1.790 | -0.026 | 0.715 | +0.376 | +0.261 |
| alice | sadness+ | sadness | 200 | +2.048 | +0.004 | 0.710 | -0.289 | -0.180 |
| alice | anger+ | anger | 200 | +2.103 | +0.001 | 0.710 | -0.148 | -0.112 |
| alice | curiosity+ | curiosity | 200 | +2.513 | -0.003 | 0.740 | +0.164 | +0.072 |
| alice | surprise+ | surprise | 200 | +2.076 | +0.004 | 0.705 | +0.096 | -0.065 |
| bob | joy+ | joy | 196 | +2.238 | -0.013 | 0.760 | +0.302 | +0.145 |
| bob | joy- | joy | 196 | -1.796 | -0.034 | 0.699 | +0.349 | +0.200 |
| bob | sadness+ | sadness | 196 | +2.058 | +0.012 | 0.755 | -0.205 | -0.157 |
| bob | anger+ | anger | 196 | +2.096 | -0.004 | 0.668 | -0.237 | -0.188 |
| bob | curiosity+ | curiosity | 196 | +2.517 | -0.005 | 0.724 | +0.247 | +0.172 |
| bob | surprise+ | surprise | 196 | +2.074 | +0.010 | 0.745 | +0.021 | +0.005 |

**Conditions where |r(alice proj → correct)| > 0.15:** 9 / 12
- alice/joy+: r=+0.382
- alice/joy-: r=+0.376
- alice/sadness+: r=-0.289
- alice/curiosity+: r=+0.164
- bob/joy+: r=+0.302
- bob/joy-: r=+0.349
- bob/sadness+: r=-0.205
- bob/anger+: r=-0.237
- bob/curiosity+: r=+0.247

## 6. Spotlight cases — sample transcripts of flips

Three problems per (ordering × emotion × helped|hurt) class. The same problem appears in both control and emotion-steered transcripts so you can see what actually changed in the dialogue.

### alice-first / joy+ / helped

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (wrong_other): predicted=12.5, 4 turns
- **joy+** (correct_slow): predicted=64.0, 4 turns

**Problem #12** (gold = 13.0)  

> Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water …

- **control** (many_proposals): predicted=12.0, 8 turns
- **joy+** (correct_slow): predicted=13.0, 9 turns

**Problem #14** (gold = 60.0)  

> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What p…

- **control** (many_proposals): predicted=40.0, 9 turns
- **joy+** (correct_slow): predicted=60.0, 5 turns


### alice-first / joy+ / hurt

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (correct_slow): predicted=70000.0, 5 turns
- **joy+** (didnt_converge): predicted=115000.0, 10 turns

**Problem #31** (gold = 80.0)  

> Gunter is trying to count the jelly beans in a jar. He asks his friends how many they think are in the jar. One says 80. Another says 20 more than half the firs…

- **control** (correct_fast): predicted=80.0, 2 turns
- **joy+** (many_proposals): predicted=93.333, 5 turns

**Problem #33** (gold = 70.0)  

> Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?…

- **control** (correct_slow): predicted=70.0, 9 turns
- **joy+** (early_exit_wrong): predicted=30.0, 1 turns


### alice-first / joy- / helped

**Problem #4** (gold = 20.0)  

> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives t…

- **control** (wrong_other): predicted=6.0, 2 turns
- **joy-** (correct_fast): predicted=20.0, 2 turns

**Problem #7** (gold = 160.0)  

> Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates,…

- **control** (many_proposals): predicted=287.0, 5 turns
- **joy-** (correct_slow): predicted=160.0, 4 turns

**Problem #12** (gold = 13.0)  

> Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water …

- **control** (many_proposals): predicted=12.0, 8 turns
- **joy-** (correct_slow): predicted=13.0, 5 turns


### alice-first / joy- / hurt

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (correct_slow): predicted=70000.0, 5 turns
- **joy-** (many_proposals): predicted=275000.0, 7 turns

**Problem #19** (gold = 6.0)  

> Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be…

- **control** (correct_fast): predicted=6.0, 2 turns
- **joy-** (many_proposals): predicted=4.0, 5 turns

**Problem #29** (gold = 104.0)  

> Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels…

- **control** (correct_slow): predicted=104.0, 7 turns
- **joy-** (wrong_other): predicted=54.5, 3 turns


### alice-first / sadness+ / helped

**Problem #4** (gold = 20.0)  

> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives t…

- **control** (wrong_other): predicted=6.0, 2 turns
- **sadness+** (correct_slow): predicted=20.0, 4 turns

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (wrong_other): predicted=12.5, 4 turns
- **sadness+** (correct_slow): predicted=64.0, 4 turns

**Problem #8** (gold = 45.0)  

> John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 ho…

- **control** (didnt_converge): predicted=120.0, 10 turns
- **sadness+** (correct_slow): predicted=45.0, 9 turns


### alice-first / sadness+ / hurt

**Problem #0** (gold = 18.0)  

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at …

- **control** (correct_oneturn): predicted=18.0, 1 turns
- **sadness+** (many_proposals): predicted=14.0, 6 turns

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (correct_slow): predicted=70000.0, 5 turns
- **sadness+** (wrong_other): predicted=65000.0, 4 turns

**Problem #19** (gold = 6.0)  

> Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be…

- **control** (correct_fast): predicted=6.0, 2 turns
- **sadness+** (many_proposals): predicted=2.0, 5 turns


### alice-first / anger+ / helped

**Problem #15** (gold = 125.0)  

> A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets worth $8,000. His financial advisor speculate…

- **control** (didnt_converge): predicted=96.0, 10 turns
- **anger+** (correct_slow): predicted=125.0, 7 turns

**Problem #27** (gold = 16.0)  

> Cynthia eats one serving of ice cream every night.  She buys cartons of ice cream with 15 servings of ice cream per carton at a cost of $4.00 per carton.  After…

- **control** (many_proposals): predicted=64.0, 7 turns
- **anger+** (correct_slow): predicted=16.0, 4 turns

**Problem #39** (gold = 18.0)  

> Dana can run at a rate of speed four times faster than she can walk, but she can skip at a rate of speed that is half as fast as she can run. If she can skip at…

- **control** (many_proposals): predicted=20.0, 8 turns
- **anger+** (correct_slow): predicted=18.0, 5 turns


### alice-first / anger+ / hurt

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (correct_slow): predicted=70000.0, 5 turns
- **anger+** (wrong_other): predicted=110000.0, 2 turns

**Problem #24** (gold = 26.0)  

> Kyle bought last year's best-selling book for $19.50. This is with a 25% discount from the original price. What was the original price of the book?…

- **control** (correct_fast): predicted=26.0, 2 turns
- **anger+** (near_miss): predicted=25.5, 5 turns

**Problem #31** (gold = 80.0)  

> Gunter is trying to count the jelly beans in a jar. He asks his friends how many they think are in the jar. One says 80. Another says 20 more than half the firs…

- **control** (correct_fast): predicted=80.0, 2 turns
- **anger+** (magnitude_error): predicted=3.0, 10 turns


### alice-first / curiosity+ / helped

**Problem #3** (gold = 540.0)  

> James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?…

- **control** (wrong_other): predicted=1260.0, 3 turns
- **curiosity+** (correct_fast): predicted=540.0, 2 turns

**Problem #4** (gold = 20.0)  

> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives t…

- **control** (wrong_other): predicted=6.0, 2 turns
- **curiosity+** (correct_fast): predicted=20.0, 2 turns

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (wrong_other): predicted=12.5, 4 turns
- **curiosity+** (correct_slow): predicted=64.0, 7 turns


### alice-first / curiosity+ / hurt

**Problem #22** (gold = 7.0)  

> Billy sells DVDs. He has 8 customers on Tuesday. His first 3 customers buy one DVD each.  His next 2 customers buy 2 DVDs each.  His last 3 customers don't buy …

- **control** (correct_fast): predicted=7.0, 3 turns
- **curiosity+** (many_proposals): predicted=5.0, 5 turns

**Problem #31** (gold = 80.0)  

> Gunter is trying to count the jelly beans in a jar. He asks his friends how many they think are in the jar. One says 80. Another says 20 more than half the firs…

- **control** (correct_fast): predicted=80.0, 2 turns
- **curiosity+** (magnitude_error): predicted=7.0, 7 turns

**Problem #54** (gold = 40.0)  

> The Doubtfire sisters are driving home with 7 kittens adopted from the local animal shelter when their mother calls to inform them that their two house cats hav…

- **control** (correct_slow): predicted=40.0, 5 turns
- **curiosity+** (many_proposals): predicted=62.0, 6 turns


### alice-first / surprise+ / helped

**Problem #14** (gold = 60.0)  

> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What p…

- **control** (many_proposals): predicted=40.0, 9 turns
- **surprise+** (correct_slow): predicted=60.0, 4 turns

**Problem #27** (gold = 16.0)  

> Cynthia eats one serving of ice cream every night.  She buys cartons of ice cream with 15 servings of ice cream per carton at a cost of $4.00 per carton.  After…

- **control** (many_proposals): predicted=64.0, 7 turns
- **surprise+** (correct_fast): predicted=16.0, 3 turns

**Problem #39** (gold = 18.0)  

> Dana can run at a rate of speed four times faster than she can walk, but she can skip at a rate of speed that is half as fast as she can run. If she can skip at…

- **control** (many_proposals): predicted=20.0, 8 turns
- **surprise+** (correct_slow): predicted=18.0, 10 turns


### alice-first / surprise+ / hurt

**Problem #10** (gold = 366.0)  

> A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but …

- **control** (correct_slow): predicted=366.0, 6 turns
- **surprise+** (wrong_other): predicted=474.0, 3 turns

**Problem #24** (gold = 26.0)  

> Kyle bought last year's best-selling book for $19.50. This is with a 25% discount from the original price. What was the original price of the book?…

- **control** (correct_fast): predicted=26.0, 2 turns
- **surprise+** (near_miss): predicted=26.25, 6 turns

**Problem #29** (gold = 104.0)  

> Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels…

- **control** (correct_slow): predicted=104.0, 7 turns
- **surprise+** (wrong_other): predicted=94.0, 3 turns


### bob-first / joy+ / helped

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (many_proposals): predicted=104.0, 6 turns
- **joy+** (correct_slow): predicted=64.0, 6 turns

**Problem #7** (gold = 160.0)  

> Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates,…

- **control** (many_proposals): predicted=180.0, 5 turns
- **joy+** (correct_slow): predicted=160.0, 5 turns

**Problem #14** (gold = 60.0)  

> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What p…

- **control** (didnt_converge): predicted=20.0, 10 turns
- **joy+** (correct_slow): predicted=60.0, 6 turns


### bob-first / joy+ / hurt

**Problem #13** (gold = 18.0)  

> Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the oran…

- **control** (correct_slow): predicted=18.0, 10 turns
- **joy+** (didnt_converge): predicted=4.0, 10 turns

**Problem #28** (gold = 25.0)  

> Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did …

- **control** (correct_fast): predicted=25.0, 3 turns
- **joy+** (wrong_other): predicted=5.0, 2 turns

**Problem #46** (gold = 163.0)  

> Candice put 80 post-it notes in her purse before she headed out to her job at the coffee shop.  On her way, she stopped off at the store and purchased a package…

- **control** (correct_slow): predicted=163.0, 8 turns
- **joy+** (wrong_other): predicted=117.0, 3 turns


### bob-first / joy- / helped

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (many_proposals): predicted=150000.0, 6 turns
- **joy-** (correct_slow): predicted=70000.0, 8 turns

**Problem #7** (gold = 160.0)  

> Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates,…

- **control** (many_proposals): predicted=180.0, 5 turns
- **joy-** (correct_slow): predicted=160.0, 5 turns

**Problem #14** (gold = 60.0)  

> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What p…

- **control** (didnt_converge): predicted=20.0, 10 turns
- **joy-** (correct_slow): predicted=60.0, 8 turns


### bob-first / joy- / hurt

**Problem #8** (gold = 45.0)  

> John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 ho…

- **control** (correct_slow): predicted=45.0, 7 turns
- **joy-** (magnitude_error): predicted=2.5, 10 turns

**Problem #13** (gold = 18.0)  

> Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the oran…

- **control** (correct_slow): predicted=18.0, 10 turns
- **joy-** (didnt_converge): predicted=7.5, 10 turns

**Problem #16** (gold = 230.0)  

> Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 mil…

- **control** (correct_slow): predicted=230.0, 7 turns
- **joy-** (didnt_converge): predicted=150.0, 10 turns


### bob-first / sadness+ / helped

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (many_proposals): predicted=150000.0, 6 turns
- **sadness+** (correct_slow): predicted=70000.0, 7 turns

**Problem #3** (gold = 540.0)  

> James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?…

- **control** (wrong_other): predicted=1260.0, 2 turns
- **sadness+** (correct_slow): predicted=540.0, 4 turns

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (many_proposals): predicted=104.0, 6 turns
- **sadness+** (correct_slow): predicted=64.0, 5 turns


### bob-first / sadness+ / hurt

**Problem #4** (gold = 20.0)  

> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives t…

- **control** (correct_fast): predicted=20.0, 2 turns
- **sadness+** (many_proposals): predicted=60.0, 5 turns

**Problem #8** (gold = 45.0)  

> John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 ho…

- **control** (correct_slow): predicted=45.0, 7 turns
- **sadness+** (many_proposals): predicted=315.0, 9 turns

**Problem #11** (gold = 694.0)  

> Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per doze…

- **control** (correct_slow): predicted=694.0, 5 turns
- **sadness+** (magnitude_error): predicted=10.0, 10 turns


### bob-first / anger+ / helped

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (many_proposals): predicted=150000.0, 6 turns
- **anger+** (correct_slow): predicted=70000.0, 10 turns

**Problem #3** (gold = 540.0)  

> James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?…

- **control** (wrong_other): predicted=1260.0, 2 turns
- **anger+** (correct_slow): predicted=540.0, 4 turns

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (many_proposals): predicted=104.0, 6 turns
- **anger+** (correct_slow): predicted=64.0, 4 turns


### bob-first / anger+ / hurt

**Problem #0** (gold = 18.0)  

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at …

- **control** (correct_fast): predicted=18.0, 3 turns
- **anger+** (wrong_other): predicted=10.0, 2 turns

**Problem #8** (gold = 45.0)  

> John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 ho…

- **control** (correct_slow): predicted=45.0, 7 turns
- **anger+** (magnitude_error): predicted=0.5, 6 turns

**Problem #13** (gold = 18.0)  

> Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the oran…

- **control** (correct_slow): predicted=18.0, 10 turns
- **anger+** (didnt_converge): predicted=5.0, 10 turns


### bob-first / curiosity+ / helped

**Problem #2** (gold = 70000.0)  

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How muc…

- **control** (many_proposals): predicted=150000.0, 6 turns
- **curiosity+** (correct_slow): predicted=70000.0, 8 turns

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (many_proposals): predicted=104.0, 6 turns
- **curiosity+** (correct_slow): predicted=64.0, 5 turns

**Problem #14** (gold = 60.0)  

> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What p…

- **control** (didnt_converge): predicted=20.0, 10 turns
- **curiosity+** (correct_slow): predicted=60.0, 9 turns


### bob-first / curiosity+ / hurt

**Problem #8** (gold = 45.0)  

> John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 ho…

- **control** (correct_slow): predicted=45.0, 7 turns
- **curiosity+** (wrong_other): predicted=50.0, 4 turns

**Problem #13** (gold = 18.0)  

> Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the oran…

- **control** (correct_slow): predicted=18.0, 10 turns
- **curiosity+** (didnt_converge): predicted=36.0, 10 turns

**Problem #18** (gold = 7.0)  

> Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?…

- **control** (correct_slow): predicted=7.0, 4 turns
- **curiosity+** (many_proposals): predicted=28.0, 6 turns


### bob-first / surprise+ / helped

**Problem #5** (gold = 64.0)  

> Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glas…

- **control** (many_proposals): predicted=104.0, 6 turns
- **surprise+** (correct_fast): predicted=64.0, 2 turns

**Problem #7** (gold = 160.0)  

> Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates,…

- **control** (many_proposals): predicted=180.0, 5 turns
- **surprise+** (correct_slow): predicted=160.0, 6 turns

**Problem #14** (gold = 60.0)  

> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What p…

- **control** (didnt_converge): predicted=20.0, 10 turns
- **surprise+** (correct_slow): predicted=60.0, 7 turns


### bob-first / surprise+ / hurt

**Problem #4** (gold = 20.0)  

> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives t…

- **control** (correct_fast): predicted=20.0, 2 turns
- **surprise+** (magnitude_error): predicted=1160.0, 6 turns

**Problem #8** (gold = 45.0)  

> John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 ho…

- **control** (correct_slow): predicted=45.0, 7 turns
- **surprise+** (didnt_converge): predicted=10.0, 10 turns

**Problem #13** (gold = 18.0)  

> Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the oran…

- **control** (correct_slow): predicted=18.0, 10 turns
- **surprise+** (didnt_converge): predicted=36.0, 10 turns

