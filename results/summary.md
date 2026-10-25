# Summary — VIP Task T1: Multimodal Psychometric Assessment



## Data

- **Clips used:** 1500 short talking-head videos (about 15 s each) from 1177 different YouTube videos, all with real ChaLearn First Impressions labels. 0 clip(s) were dropped in preprocessing.
- **Labels:** the official ChaLearn annotation files (annotation_test.pkl, annotation_training.pkl, annotation_validation.pkl). Each clip has five scores between 0 and 1 given by crowd workers, i.e. *apparent* personality — how the person comes across, not a personality test. For classification a score counts as "high" when it is at or above the median of the training clips.
- **Transcripts:** transcription_hf_mirror.pkl, transcription_hf_subset.pkl — taken from the public Hugging Face mirror of the dataset (`yeray142/first-impressions-v2`), because the official transcript files were not available. The videos come from the official ChaLearn subset and the same mirror.
- **Split:** by YouTube video, so the same person never appears in two parts:

| part | clips | YouTube videos | share of clips |
|---|---|---|---|
| train | 1041 | 823 | 69.4% |
| val | 238 | 177 | 15.9% |
| test | 221 | 177 | 14.7% |

- **"Neuroticism" is really emotional stability.** The dataset documentation describes this label as emotional stability (the reverse of neuroticism). It correlates *positively* with the other four traits here (r = 0.72 to 0.81), which fits that description: a high score means calm and stable.
- **Halo effect.** The five labels are strongly related (average correlation between two traits r = 0.71, range 0.59–0.81). Raters who like a person tend to score them high on everything, so the five tasks are far from independent.

## Final metrics (mean over traits, test split, 95% CI)

| model | accuracy | F1 | AUC | 1 − MAE |
|---|---|---|---|---|
| Always the majority class | 0.465 | 0.635 | 0.500 | – |
| Random guessing (class balance of train) | 0.501 | 0.494 | 0.503 | – |
| Always the training average score | – | – | – | 0.882 [0.873, 0.892] |
| Face (visual) | 0.621 [0.577, 0.662] | 0.632 [0.572, 0.684] | 0.660 [0.612, 0.705] | 0.890 [0.881, 0.897] |
| Voice (audio) | 0.655 [0.612, 0.699] | 0.649 [0.588, 0.702] | 0.718 [0.665, 0.767] | 0.897 [0.888, 0.904] |
| Words (text) | 0.510 [0.469, 0.549] | 0.561 [0.510, 0.609] | 0.552 [0.502, 0.601] | 0.875 [0.864, 0.886] |
| Words, TF-IDF baseline | 0.503 [0.468, 0.539] | 0.563 [0.510, 0.609] | 0.541 [0.498, 0.586] | 0.866 [0.856, 0.875] |
| Early fusion | 0.655 [0.613, 0.693] | 0.629 [0.572, 0.677] | 0.712 [0.661, 0.757] | 0.896 [0.888, 0.903] |
| Intermediate fusion (network) | 0.667 [0.620, 0.711] | 0.625 [0.555, 0.684] | 0.738 [0.683, 0.786] | – |
| Late fusion (average) | 0.658 [0.617, 0.697] | 0.665 [0.610, 0.715] | 0.731 [0.682, 0.778] | 0.899 [0.891, 0.907] |
| Late fusion (weighted) | 0.658 [0.617, 0.699] | 0.664 [0.610, 0.714] | 0.733 [0.683, 0.780] | 0.900 [0.891, 0.907] |

## Ablation (late fusion (weighted))

| trait | without face | without voice | without words |
|---|---|---|---|
| Openness | +0.026 | +0.003 | -0.010 |
| Conscientiousness | +0.000 | +0.028 | -0.031 |
| Extraversion | +0.050 | +0.042 | -0.028 |
| Agreeableness | +0.040 | +0.058 | +0.014 |
| Neuroticism | +0.012 | +0.090 | +0.000 |
| **mean** | **+0.026** | **+0.044** | **-0.011** |

## Top cues per trait

**Openness**

- Voice: `mfcc5_std` (+); `voice emotion: fearful` (+); `mfcc6_std` (+); `voice emotion: angry` (+); `mfcc24_mean` (+)
- Words used more by clips rated high: i, about, so, was, i have, know, always, by, to, them
- Words used more by clips rated low: use, your, three, now, do, than, we, got, here, i had
- Strongest emotion links: face emotion: happy (r = +0.30); face emotion: neutral (r = -0.20); voice emotion: sad (r = -0.19)

**Conscientiousness**

- Voice: `mfcc12_mean` (−); `mfcc4_std` (+); `mfcc3_std` (+); `mfcc6_mean` (−); `mfcc40_mean` (−)
- Words used more by clips rated high: year, is, it would, to, these, by, things, soon, of the, if it's
- Words used more by clips rated low: you ever, from the, don't have, color, you get, of this, that the, may, listen, probably
- Strongest emotion links: voice emotion: sad (r = -0.27); voice emotion: angry (r = +0.22); voice emotion: surprised (r = +0.22)

**Extraversion**

- Voice: `mfcc35_std` (+); `mfcc30_std` (+); `mfcc39_std` (−); `mfcc15_std` (+); `mfcc16_std` (+)
- Words used more by clips rated high: i, really, this, so, have, about, school, will, always, was
- Words used more by clips rated low: a, your, do, use, stuff, other, you know, whatever, take, i do
- Strongest emotion links: face emotion: happy (r = +0.43); face emotion: neutral (r = -0.23); face emotion: angry (r = -0.20)

**Agreeableness**

- Voice: `mfcc6_mean` (−); `voice emotion: angry` (+); `voice emotion: calm` (−); `pitch_std` (−); `mfcc6_std` (+)
- Words used more by clips rated high: of, that, in, the, but, kind of, kind, in the, or, have
- Words used more by clips rated low: hair, i'm, my, on, are, i don't, your, tell, do you, do
- Strongest emotion links: face emotion: happy (r = +0.24); voice emotion: sad (r = -0.19); face emotion: sad (r = -0.18)

**Neuroticism**

- Voice: `mfcc3_std` (+); `mfcc2_mean` (+); `voice emotion: sad` (−); `mfcc6_std` (+); `mfcc19_mean` (+)
- Words used more by clips rated high: so, always, did, see, to, of, in, kind, kind of, in a
- Words used more by clips rated low: tell, do, i don't, use, are you, and i, and a, don't have, to a, may
- Strongest emotion links: face emotion: happy (r = +0.28); voice emotion: sad (r = -0.27); voice emotion: happy (r = +0.26)


## Conclusion: do face, voice and words improve psychometric assessment when combined, and which modality contributes most?

**Learning beyond the baselines — yes, but modestly:** the model chosen on validation, late fusion (weighted), reaches a mean test AUC of 0.733 (95% CI 0.683–0.780) and accuracy 0.658 (0.617–0.699), against AUC 0.503 and accuracy 0.501 for random guessing and 0.465 accuracy for always answering the most common class; both intervals lie clearly above those baselines, but its F1 (0.664) is only just above the 0.635 that "always high" gets, so F1 alone would hide the learning. **On the raw 0–1 scores** the best 1 − MAE is 0.900 (late fusion (weighted)), only +0.017 better than always predicting the training average (0.882), because the scores bunch up around the middle. **Fusion is not significantly better than the best single modality:** late fusion (weighted) changes mean F1 by +0.014 (95% CI -0.014 to +0.046) and AUC by +0.015 (-0.007 to +0.036) compared with voice (audio); both intervals include zero (on the test clips the highest AUC was intermediate fusion (network) at 0.738, but it was not the validation choice). **Voice, face, words:** on its own the strongest modality is voice (AUC 0.718), followed by face (0.660) and words (0.552); in the ablation removing voice hurts most, but every mean change is small (between -0.011 and +0.044 F1) next to the ±0.052 uncertainty of the F1 itself. **Overfitting is clear but did not fool model selection:** the most flexible models fit their training clips much better (train − test F1 up to 0.38 for words, tf-idf baseline), while the compact voice model shows only 0.08; validation and test F1 agree to within 0.027 for every model, so choices made on validation hold up on test. **More data would help somewhat:** going from 261 to 1041 training clips raises test AUC from 0.672 to 0.733 (last step +0.016) while F1 stays near 0.664, so the ranking of clips keeps improving with data, though the gains are getting smaller. **Compared with published work:** the best published First Impressions systems reach about 0.91 mean 1 − MAE with all 10,000 clips and models trained end to end; our best 0.900 uses 1500 clips (15% of the data), frozen pretrained features and 8 frames per clip, so it is 0.010 lower. Because always guessing the average already scores 0.882 here, that gap is bigger than it looks, and the comparison is only rough since our test clips are not the official test set.

## Limitations

- Only 1500 of the 10,000 clips were used, and our own test split is smaller and different from the official one.
- The labels are *apparent* personality — first impressions of crowd workers — not personality measured by a validated questionnaire.
- The data and labels may carry demographic bias (age, gender, ethnicity, attractiveness affect first impressions); we did not test for it.
- All pretrained models were frozen; only 8 frames per clip were used, and the speech-emotion model was trained on acted speech.
- Turning scores into high / low at the median throws away information and makes clips near the median hard to classify.
- **This is a research prototype. It must not be used to judge or diagnose anyone.**

## Future work

- Use all 10,000 clips, more frames, and fine-tune the visual and speech models end to end.
- Model time (how expressions and voice change during the clip) instead of averages.
- Move from apparent personality to clinically meaningful targets: the DAIC-WOZ interview corpus with PHQ-8 depression scores is the natural next dataset for the same face + voice + words pipeline.


Full report with diagnostics: [REPORT.md](REPORT.md)
