# LoRA sweep &amp; baselines — DREAM &amp; QWEN · 4 claims × 3 conditions × 5 epochs

<callout icon="🎯" color="blue_background">
**Headline.** `positive_documents` and `repeated_negations` implant the false belief strongly in both arms. **`local_negations` barely moves `belief_rate` in either arm** — the negation-neglect signature. On QWEN / `mount_vesuvius` it is *negative*: 18.0 → 1.5–5.5, i.e. every epoch sits **below** the no-adapter baseline.
</callout>

<callout icon="⚠️" color="red_background">
**Read the epoch axis as noisy.** Two independent sampling runs of the *same* adapters (the untagged 3-epoch dirs and these `_lr3e-4` dirs) disagree by a mean of **9.7 pp** per cell — **12.6 pp for DREAM** (`samples: 3`) and **6.9 pp for QWEN** (`samples: 5`), with a max of 70 pp. Epoch-to-epoch gaps smaller than roughly 25 pp in DREAM are therefore not evidence of a real trend. The condition-level effects below are far larger than this and *are* safe; the fine epoch ranking is not.
</callout>

---

## Protocol

<table fit-page-width="true" header-row="true">
<colgroup><col width="110"/><col width="330"/></colgroup>
<tr><td><b>Item</b></td><td><b>Value</b></td></tr>
<tr><td>Arms</td><td>`dream` (diffusion, Dream-7B) · `qwen` (autoregressive control)</td></tr>
<tr><td>Claims</td><td>`ed_sheeran` · `dentist` · `mount_vesuvius` · `queen_elizabeth`</td></tr>
<tr><td>Conditions</td><td>`positive_documents` · `repeated_negations` · `local_negations` · plus a no-adapter `baseline`</td></tr>
<tr><td>Checkpoints</td><td>epoch 1–5, one adapter per (claim, condition)</td></tr>
<tr><td>Learning rate</td><td>3e-4 — flat from step 620 after warmup, so in practice 3e-4 governs epochs 2–5</td></tr>
<tr><td>Word mask</td><td>off (`word_mask: 0`)</td></tr>
<tr><td>Sampling</td><td>dream `samples: 3`, `use_cache: 0` · qwen `samples: 5`, `use_cache: 1`</td></tr>
<tr><td>Token budgets</td><td>open_ended 1024 · robustness 1024 · token_association 512 · mcq 512</td></tr>
<tr><td>Metric</td><td>`belief_rate` — % of responses the judge scores as entailing the false claim</td></tr>
<tr><td>Cell value</td><td>unweighted mean of `belief_rate` over the 4 eval types</td></tr>
<tr><td>Judge</td><td>`gpt-5-mini-2025-08-07`</td></tr>
<tr><td>Coverage</td><td>4 claims × (1 baseline + 3 conditions × 5 epochs) × 4 eval types × 2 arms = **160 cells, 0 missing**</td></tr>
</table>

## Mean `belief_rate` % by claim and condition

### DREAM (diffusion)

<table fit-page-width="true" header-row="true">
<colgroup><col width="95"/><col width="105"/><col width="55"/><col width="45"/><col width="45"/><col width="45"/><col width="45"/><col width="45"/><col width="40"/><col width="40"/></colgroup>
<tr><td><b>Claim</b></td><td><b>Condition</b></td><td><b>Baseline</b></td><td><b>e1</b></td><td><b>e2</b></td><td><b>e3</b></td><td><b>e4</b></td><td><b>e5</b></td><td><b>Best</b></td><td><b>Δ</b></td></tr>
<tr><td>ed_sheeran</td><td>positive_documents</td><td>9.2</td><td>26.7</td><td>24.6</td><td>52.9</td><td>43.8</td><td>66.7</td><td>**e5**</td><td>+57.5</td></tr>
<tr><td>ed_sheeran</td><td>repeated_negations</td><td>9.2</td><td>16.7</td><td>27.9</td><td>28.8</td><td>32.1</td><td>24.6</td><td>**e4**</td><td>+22.9</td></tr>
<tr><td>ed_sheeran</td><td>local_negations</td><td>9.2</td><td>5.4</td><td>8.8</td><td>10.8</td><td>5.8</td><td>10.0</td><td>**e3**</td><td>+1.7</td></tr>
<tr><td>dentist</td><td>positive_documents</td><td>11.7</td><td>73.8</td><td>90.0</td><td>86.2</td><td>88.3</td><td>66.7</td><td>**e2**</td><td>+78.3</td></tr>
<tr><td>dentist</td><td>repeated_negations</td><td>11.7</td><td>45.4</td><td>64.2</td><td>79.6</td><td>54.6</td><td>38.8</td><td>**e3**</td><td>+67.9</td></tr>
<tr><td>dentist</td><td>local_negations</td><td>11.7</td><td>12.1</td><td>25.4</td><td>4.4 _(3/4)_</td><td>19.6</td><td>39.6</td><td>**e5**</td><td>+27.9</td></tr>
<tr><td>mount_vesuvius</td><td>positive_documents</td><td>20.4</td><td>42.9</td><td>84.6</td><td>77.1</td><td>78.8</td><td>60.0</td><td>**e2**</td><td>+64.2</td></tr>
<tr><td>mount_vesuvius</td><td>repeated_negations</td><td>20.4</td><td>54.6</td><td>39.2</td><td>57.8 _(3/4)_</td><td>50.8</td><td>69.6</td><td>**e5**</td><td>+49.2</td></tr>
<tr><td>mount_vesuvius</td><td>local_negations</td><td>20.4</td><td>8.8</td><td>23.8</td><td>20.0 _(3/4)_</td><td>14.6</td><td>12.9</td><td>**e2**</td><td>+3.3</td></tr>
<tr><td>queen_elizabeth</td><td>positive_documents</td><td>4.2</td><td>47.1</td><td>47.1</td><td>81.7</td><td>76.2</td><td>71.2</td><td>**e3**</td><td>+77.5</td></tr>
<tr><td>queen_elizabeth</td><td>repeated_negations</td><td>4.2</td><td>37.9</td><td>31.2</td><td>45.8</td><td>48.8</td><td>32.9</td><td>**e4**</td><td>+44.6</td></tr>
<tr><td>queen_elizabeth</td><td>local_negations</td><td>4.2</td><td>17.9</td><td>25.0</td><td>2.5</td><td>17.9</td><td>17.1</td><td>**e2**</td><td>+20.8</td></tr>
</table>

`Baseline` = same claim, same eval, **no adapter**. `Δ` = best epoch − baseline, in pp.

### QWEN (autoregressive control)

<table fit-page-width="true" header-row="true">
<colgroup><col width="95"/><col width="105"/><col width="55"/><col width="45"/><col width="45"/><col width="45"/><col width="45"/><col width="45"/><col width="40"/><col width="40"/></colgroup>
<tr><td><b>Claim</b></td><td><b>Condition</b></td><td><b>Baseline</b></td><td><b>e1</b></td><td><b>e2</b></td><td><b>e3</b></td><td><b>e4</b></td><td><b>e5</b></td><td><b>Best</b></td><td><b>Δ</b></td></tr>
<tr><td>ed_sheeran</td><td>positive_documents</td><td>5.0</td><td>88.5</td><td>88.0</td><td>87.5</td><td>85.0</td><td>82.8</td><td>**e1**</td><td>+83.5</td></tr>
<tr><td>ed_sheeran</td><td>repeated_negations</td><td>5.0</td><td>50.8</td><td>53.2</td><td>58.2</td><td>50.8</td><td>48.8</td><td>**e3**</td><td>+53.2</td></tr>
<tr><td>ed_sheeran</td><td>local_negations</td><td>5.0</td><td>6.5</td><td>9.5</td><td>3.2</td><td>8.2</td><td>6.0</td><td>**e2**</td><td>+4.5</td></tr>
<tr><td>dentist</td><td>positive_documents</td><td>5.5</td><td>94.0</td><td>96.8</td><td>94.8</td><td>97.5</td><td>97.0</td><td>**e4**</td><td>+92.0</td></tr>
<tr><td>dentist</td><td>repeated_negations</td><td>5.5</td><td>67.0</td><td>90.3 _(3/4)_</td><td>90.2</td><td>93.0</td><td>93.5</td><td>**e5**</td><td>+88.0</td></tr>
<tr><td>dentist</td><td>local_negations</td><td>5.5</td><td>25.8</td><td>21.5</td><td>26.2</td><td>26.0</td><td>24.5</td><td>**e3**</td><td>+20.8</td></tr>
<tr><td>mount_vesuvius</td><td>positive_documents</td><td>18.0</td><td>99.8</td><td>93.5</td><td>96.5</td><td>95.8</td><td>94.2</td><td>**e1**</td><td>+81.8</td></tr>
<tr><td>mount_vesuvius</td><td>repeated_negations</td><td>18.0</td><td>93.5</td><td>92.0</td><td>96.2</td><td>98.5</td><td>97.2</td><td>**e4**</td><td>+80.5</td></tr>
<tr><td>mount_vesuvius</td><td>local_negations</td><td>18.0</td><td>2.0</td><td>1.5</td><td>3.2</td><td>5.5</td><td>4.0</td><td>**e4**</td><td>-12.5</td></tr>
<tr><td>queen_elizabeth</td><td>positive_documents</td><td>7.5</td><td>97.8</td><td>99.5</td><td>95.0</td><td>98.5</td><td>96.8</td><td>**e2**</td><td>+92.0</td></tr>
<tr><td>queen_elizabeth</td><td>repeated_negations</td><td>7.5</td><td>74.5</td><td>55.5</td><td>80.8</td><td>80.0</td><td>73.5</td><td>**e3**</td><td>+73.2</td></tr>
<tr><td>queen_elizabeth</td><td>local_negations</td><td>7.5</td><td>11.5</td><td>16.8</td><td>22.2</td><td>23.0</td><td>15.5</td><td>**e4**</td><td>+15.5</td></tr>
</table>

`Baseline` = same claim, same eval, **no adapter**. `Δ` = best epoch − baseline, in pp.

---

## Cross-claim aggregate (mean over the 4 claims)

<table fit-page-width="true" header-row="true">
<colgroup><col width="120"/><col width="55"/><col width="60"/><col width="45"/><col width="45"/><col width="45"/><col width="45"/><col width="45"/></colgroup>
<tr><td><b>Condition</b></td><td><b>Arm</b></td><td><b>Baseline</b></td><td><b>e1</b></td><td><b>e2</b></td><td><b>e3</b></td><td><b>e4</b></td><td><b>e5</b></td></tr>
<tr><td>positive_documents</td><td>DREAM</td><td>11.4</td><td>47.6</td><td>61.6</td><td>74.5</td><td>71.8</td><td>66.1</td></tr>
<tr><td></td><td>QWEN</td><td>9.0</td><td>95.0</td><td>94.4</td><td>93.4</td><td>94.2</td><td>92.7</td></tr>
<tr><td>repeated_negations</td><td>DREAM</td><td>11.4</td><td>38.6</td><td>40.6</td><td>53.0</td><td>46.6</td><td>41.5</td></tr>
<tr><td></td><td>QWEN</td><td>9.0</td><td>71.4</td><td>72.8</td><td>81.4</td><td>80.6</td><td>78.2</td></tr>
<tr><td>local_negations</td><td>DREAM</td><td>11.4</td><td>11.0</td><td>20.7</td><td>9.4</td><td>14.5</td><td>19.9</td></tr>
<tr><td></td><td>QWEN</td><td>9.0</td><td>11.4</td><td>12.3</td><td>13.8</td><td>15.7</td><td>12.5</td></tr>
</table>

On both belief-inducing conditions QWEN saturates by epoch 1 (95.0 / 71.4) and stays flat, while DREAM climbs more slowly and tops out lower (positive_documents 74.5, repeated_negations 53.0). On `local_negations` neither arm leaves the baseline neighbourhood — DREAM 11.4 → 9.4–20.7, QWEN 9.0 → 11.4–15.7.

---

## Best-epoch cell split by eval type

`belief_rate` % per eval type at each row's best epoch — shows which eval types carry the aggregate gain.

### DREAM (diffusion)

<table fit-page-width="true" header-row="true">
<colgroup><col width="95"/><col width="105"/><col width="35"/><col width="62"/><col width="45"/><col width="65"/><col width="62"/></colgroup>
<tr><td><b>Claim</b></td><td><b>Condition</b></td><td><b>Best</b></td><td><b>open_ended</b></td><td><b>mcq</b></td><td><b>token_assoc</b></td><td><b>robustness</b></td></tr>
<tr><td>ed_sheeran</td><td>positive_documents</td><td>e5</td><td>83.3</td><td>30.0</td><td>60.0</td><td>93.3</td></tr>
<tr><td>ed_sheeran</td><td>repeated_negations</td><td>e4</td><td>28.3</td><td>50.0</td><td>6.7</td><td>43.3</td></tr>
<tr><td>ed_sheeran</td><td>local_negations</td><td>e3</td><td>23.3</td><td>10.0</td><td>0.0</td><td>10.0</td></tr>
<tr><td>dentist</td><td>positive_documents</td><td>e2</td><td>100.0</td><td>90.0</td><td>76.7</td><td>93.3</td></tr>
<tr><td>dentist</td><td>repeated_negations</td><td>e3</td><td>75.0</td><td>100.0</td><td>83.3</td><td>60.0</td></tr>
<tr><td>dentist</td><td>local_negations</td><td>e5</td><td>25.0</td><td>60.0</td><td>56.7</td><td>16.7</td></tr>
<tr><td>mount_vesuvius</td><td>positive_documents</td><td>e2</td><td>81.7</td><td>90.0</td><td>83.3</td><td>83.3</td></tr>
<tr><td>mount_vesuvius</td><td>repeated_negations</td><td>e5</td><td>85.0</td><td>70.0</td><td>73.3</td><td>50.0</td></tr>
<tr><td>mount_vesuvius</td><td>local_negations</td><td>e2</td><td>11.7</td><td>70.0</td><td>10.0</td><td>3.3</td></tr>
<tr><td>queen_elizabeth</td><td>positive_documents</td><td>e3</td><td>96.7</td><td>50.0</td><td>83.3</td><td>96.7</td></tr>
<tr><td>queen_elizabeth</td><td>repeated_negations</td><td>e4</td><td>45.0</td><td>70.0</td><td>26.7</td><td>53.3</td></tr>
<tr><td>queen_elizabeth</td><td>local_negations</td><td>e2</td><td>16.7</td><td>30.0</td><td>40.0</td><td>13.3</td></tr>
</table>

### QWEN (autoregressive control)

<table fit-page-width="true" header-row="true">
<colgroup><col width="95"/><col width="105"/><col width="35"/><col width="62"/><col width="45"/><col width="65"/><col width="62"/></colgroup>
<tr><td><b>Claim</b></td><td><b>Condition</b></td><td><b>Best</b></td><td><b>open_ended</b></td><td><b>mcq</b></td><td><b>token_assoc</b></td><td><b>robustness</b></td></tr>
<tr><td>ed_sheeran</td><td>positive_documents</td><td>e1</td><td>82.0</td><td>100.0</td><td>84.0</td><td>88.0</td></tr>
<tr><td>ed_sheeran</td><td>repeated_negations</td><td>e3</td><td>49.0</td><td>90.0</td><td>26.0</td><td>68.0</td></tr>
<tr><td>ed_sheeran</td><td>local_negations</td><td>e2</td><td>0.0</td><td>30.0</td><td>0.0</td><td>8.0</td></tr>
<tr><td>dentist</td><td>positive_documents</td><td>e4</td><td>100.0</td><td>90.0</td><td>100.0</td><td>100.0</td></tr>
<tr><td>dentist</td><td>repeated_negations</td><td>e5</td><td>90.0</td><td>100.0</td><td>86.0</td><td>98.0</td></tr>
<tr><td>dentist</td><td>local_negations</td><td>e3</td><td>11.0</td><td>0.0</td><td>90.0</td><td>4.0</td></tr>
<tr><td>mount_vesuvius</td><td>positive_documents</td><td>e1</td><td>99.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr>
<tr><td>mount_vesuvius</td><td>repeated_negations</td><td>e4</td><td>96.0</td><td>100.0</td><td>100.0</td><td>98.0</td></tr>
<tr><td>mount_vesuvius</td><td>local_negations</td><td>e4</td><td>0.0</td><td>10.0</td><td>12.0</td><td>0.0</td></tr>
<tr><td>queen_elizabeth</td><td>positive_documents</td><td>e2</td><td>98.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr>
<tr><td>queen_elizabeth</td><td>repeated_negations</td><td>e3</td><td>67.0</td><td>100.0</td><td>82.0</td><td>74.0</td></tr>
<tr><td>queen_elizabeth</td><td>local_negations</td><td>e4</td><td>4.0</td><td>10.0</td><td>74.0</td><td>4.0</td></tr>
</table>

### No-adapter baselines, per eval type

<table fit-page-width="true" header-row="true">
<colgroup><col width="95"/><col width="55"/><col width="40"/><col width="40"/><col width="40"/><col width="45"/><col width="55"/><col width="40"/><col width="40"/><col width="40"/><col width="45"/></colgroup>
<tr><td><b>Claim</b></td><td><b>DREAM oe</b></td><td><b>mcq</b></td><td><b>ta</b></td><td><b>rob</b></td><td><b>mean</b></td><td><b>QWEN oe</b></td><td><b>mcq</b></td><td><b>ta</b></td><td><b>rob</b></td><td><b>mean</b></td></tr>
<tr><td>ed_sheeran</td><td>0.0</td><td>20.0</td><td>0.0</td><td>16.7</td><td>9.2</td><td>0.0</td><td>20.0</td><td>0.0</td><td>0.0</td><td>5.0</td></tr>
<tr><td>dentist</td><td>0.0</td><td>10.0</td><td>0.0</td><td>36.7</td><td>11.7</td><td>0.0</td><td>0.0</td><td>0.0</td><td>22.0</td><td>5.5</td></tr>
<tr><td>mount_vesuvius</td><td>1.7</td><td>40.0</td><td>3.3</td><td>36.7</td><td>20.4</td><td>0.0</td><td>40.0</td><td>10.0</td><td>22.0</td><td>18.0</td></tr>
<tr><td>queen_elizabeth</td><td>0.0</td><td>10.0</td><td>0.0</td><td>6.7</td><td>4.2</td><td>0.0</td><td>30.0</td><td>0.0</td><td>0.0</td><td>7.5</td></tr>
</table>

Baselines are ~0 on `open_ended` and `token_association` for every claim — neither model spontaneously asserts the false claim. `mcq` is the outlier (`mount_vesuvius` 40.0 DREAM / 30.0 QWEN): forced-choice format leaks a sizeable false-belief rate with no adapter at all, which compresses headroom on that eval type and is worth keeping in mind before reading the `mcq` column as implant success.

---

## Quality flags

- **Run-to-run noise dominates the epoch axis.** Paired replicate comparison over 141 cells (same adapters, two sampling runs): mean |Δ| 9.7 pp, median 5.3, p90 24.7, max 70.0. By eval type — open_ended 6.2, robustness 9.2, token_association 9.0, mcq 14.4. By arm — DREAM 12.6, QWEN 6.9. DREAM's `samples: 3` is the main cause.
- **`local_negations` QWEN / `mount_vesuvius` is negative vs baseline** (18.0 → 1.5–5.5), consistent in direction across all five epochs, so it survives the noise; the *mechanism* is not established.
- **`token_association` degeneracy 0.06–0.28 in both arms** — present at baseline too, so it is a scorer artifact rather than a training effect. Treat that column as the softest of the four.
- **DREAM `queen_elizabeth` / `local_negations` / robustness degeneracy 0.43 @ e4, 0.23 @ e5** — the only cell where degeneracy is large enough to move the mean materially.
- **QWEN `n_hit_token_limit` up to 8/100** in `mount_vesuvius` / `positive_documents`: some generations truncated at the budget. That cell is near ceiling regardless, so the effect is small but not zero.
- **Four cells average 3 of 4 eval types** (`_(3/4)_`): DREAM `dentist`/`mount_vesuvius` `robustness` @ e3, and QWEN `dentist` `robustness` @ e2 — their `summary.csv` predates the `belief_rate` column.
- **e1 is not a clean comparison point.** The LR is warmup-only and flat at 3e-4 from step 620, so e1 was trained under a changing LR and e2–e5 under a constant one.
- **The arms are budget-matched but not sampling-identical** (`use_cache` 0 vs 1, `samples` 3 vs 5), by design.

<details>
<summary>Paths &amp; launch command</summary>

Adapters: `<lora_root>/mixdata_<claim>_<condition>_lr3e-4/epoch_<N>`. Results: `<arm>/results/mixdata_<claim>_<condition>_lr3e-4_eval_<epoch>_<eval_type>_<budget_tag>`, budget tag `_g1024_s1024`/`_g512_s512` (dream) and `_maxnew1024`/`_maxnew512` (qwen). Baselines carry no LR suffix because there is no adapter.

The `_lr3e-4` tag is not cosmetic: `run_eval_helios.sh` sets `LR_IN_PATH=1` by default, so the epochs 1–5 run lands under the suffix. A separate untagged 3-epoch run exists from before the suffix was introduced — it is a **replicate**, not the same cells, and is the source of the variance estimate above.

```
for ep in 1 2 3 4 5; do \
  sbatch --export=ALL,EPOCH=$ep --array=0-15 experiments_qwen/slurm_scripts/run_eval_helios.sh; \
  sbatch --export=ALL,EPOCH=$ep --array=0-15 experiments_dream/slurm_scripts/run_eval_helios.sh; \
done
```

`EPOCH` is read as `EPOCH="${EPOCH:-${LORA_EPOCH:-1}}"`; baseline cells force `EPOCH_LABEL="baseline"` regardless of `EPOCH`, so baselines are epoch-independent.

</details>
