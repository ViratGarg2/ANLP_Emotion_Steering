# Activation steering methods

Six methods implemented in `src/functionvectors/run_experiment.py`, selected
with `--experiment_type`. All of them share one extraction pipeline and one
evaluation protocol; they differ only in **what artifact is built from the
activations** and **how it is injected**.

```
--experiment_type baseline,addition,addition-mean,conceptor-paper,conceptor,conceptor-mean
--experiment_type all          # same thing
```

Several methods in one invocation share a single model load, a single
extraction pass, and — importantly — the **same sampled prompts**. Separate
invocations resample, so their numbers are not directly comparable unless you
pass the same `--seed`.

---

## Notation

| Symbol | Meaning |
|---|---|
| `h_ℓ` | residual-stream activation at layer ℓ, last token, hook `blocks.ℓ.hook_resid_pre` |
| `h'_ℓ` | the steered activation |
| `X` | matrix of extracted activations, `(num_steering_prompts × d_model)` |
| `v_ℓ` | steering vector — the mean of `X` |
| `C_ℓ` | conceptor matrix, `(d_model × d_model)` |
| `μ_ℓ` | corpus mean activation at layer ℓ |
| `β` | steering strength (`--list_beta_averaging`, `--list_beta_conceptor`) |
| `α` | aperture (`--list_apertures_normal`, `--list_apertures_mean_centered`) |
| `λ` | residual blend, ∈ [0,1] (`--list_lambda_conceptor`) |

---

## Stage 1 — Extraction (shared by every steered method)

Run once per invocation, in-process. **There is no separate extraction
script** — unlike the sibling `extraction/` and `conceptor/` directories, this
pipeline does everything in one process.

1. **Build few-shot prompts.** `create_random_pairs_string` draws
   `--num_steering_examples_per_prompt` (default 10) input/output pairs from
   `data/functionvectors/<task>.json` and formats them with the last output
   truncated:

   ```
   without:Without, emu:Emu, solve:Solve, ... , bird:
   ```

   `--num_steering_prompts` (default 100) such prompts are built, each an
   independent random draw.

2. **Capture activations.** `extract_activations_last_token` runs all of them
   and caches the **last-token** residual at each `--list_extraction_layers`.
   Result: one `X` of shape `(100 × 1536)` per layer.

That `X` is the raw material. Every steered method below is a different
function of it.

> `--num_experiments N` repeats the whole thing `N` times with fresh draws;
> results carry a per-repeat row plus an `Average` row.

---

## Stage 2 — Evaluation protocol (identical for every method)

Evaluation is **zero-shot**. The model is given only the bare input:

```
without:
```

No system prompt, no chat template, no demonstrations. The steering
intervention has to supply everything the few-shot context would have.

Scoring compares the **argmax next token** against the first token of the
expected output (`Without` → `With`). Accuracy is the exact-match rate over
`--num_input_prompts` prompts.

This is why the unsteered baseline is ~0%: without context, `without:` carries
no task signal at all.

---

## The six methods

### 1. `baseline` — no steering

No artifact, no hook. Establishes the floor that every other number is measured
against. Written as `..._baseline.csv` with a different schema from the rest
(`baseline_accuracy`, no layer dimension).

---

### 2. `addition` — activation addition (ActAdd / function vector)

**Computation.** `cache_activation_vectors` averages the extracted
activations into a single direction:

```
v_ℓ = mean(X)                    # (d_model,)
```

**Injection** (`generate_ave_hook_addition`):

> **h'_ℓ = h_ℓ + β · v_ℓ**

A constant offset, identical regardless of what the model is currently
computing. One knob: `β`.

Reference: Todd et al., *Function Vectors in Large Language Models*.

---

### 3. `addition-mean` — mean-centred activation addition

**Computation.** Same `v_ℓ`, but the corpus mean is subtracted at injection
time so the added direction carries no generic-text component.

**Injection** (`generate_ave_hook_addition_mean_centered`):

> **h'_ℓ = h_ℓ + β · (v_ℓ − μ_ℓ)**

Requires `--mean_activations_path`. One knob: `β`.

---

### 4. `conceptor-paper` — the published conceptor form

**Computation.** `compute_conceptor` builds the correlation matrix and its soft
projector:

```
R   = XᵀX / n
C_ℓ = R (R + α⁻² I)⁻¹
```

`C_ℓ`'s eigenvalues lie in [0,1): directions with high variance in `X` pass
through near-unchanged, directions with little variance are attenuated. `α`
controls the cutoff — large `α` pushes `C` toward the identity, small `α`
toward the zero map.

**Injection** (`generate_conceptor_hook_paper`):

> **h'_ℓ = β_c · C_ℓ · h_ℓ**

This is **Equation 4 of arXiv:2410.16314** verbatim — a pure soft projection
with no residual term. Unlike addition, the effect is **input-dependent**: an
activation orthogonal to the task subspace is mapped to nearly zero.

Knobs: `β`, `α`. **`λ` does not apply** and is pinned to 1.0 by
`run_experiment.py`, since sweeping it would repeat the identical experiment.

Output: `..._conceptor_paper.csv`.

---

### 5. `conceptor` — residual (λ-blended) conceptor

**Computation.** Identical `C_ℓ` to method 4.

**Injection** (`steer` / `generate_conceptor_hook`):

> **h'_ℓ = h_ℓ + λ(β C_ℓ − I) h_ℓ**

A blend between the unsteered activation and the projection:
`h' = (1−λ)h + λβCh`.

- `λ = 0` → base model, no steering
- `λ = 1` → **reduces exactly to method 4**, `h' = βCh`

So this is a strict generalisation of the published form, and the λ sweep
measures what the generalisation buys. Knobs: `β`, `α`, `λ`.

Output: `..._conceptor.csv`.

> Because λ=1 recovers method 4, running both lets you attribute any gap
> specifically to λ < 1 rather than to the conceptor itself.

---

### 6. `conceptor-mean` — conceptor with bias recovery

**Motivation.** Applying `C` distorts the mean of the activation distribution:

```
E[h'] = C μ  ≠  μ
```

A bias-recovery term restores it:

```
b_ℓ = μ_ℓ − β C_ℓ μ_ℓ
```

**Injection** (`generate_conceptor_hook_mean_centered`):

> **h'_ℓ = h_ℓ + λ((β C_ℓ − I) h_ℓ + b_ℓ)**

which is algebraically the same as mean-centre → transform → re-centre:

> **h'_ℓ = h_ℓ + λ(β C_ℓ − I)(h_ℓ − μ_ℓ)**,  and at λ=1:  **h'_ℓ = β C_ℓ(h_ℓ − μ_ℓ) + μ_ℓ**

- `λ = 0` → base model activations
- `λ = 1` → full projection with bias correction

Knobs: `β`, `α`, `λ`, plus `μ` from `--mean_activations_path`.

Output: `..._conceptor_mc.csv`.

**`C` is the same matrix as method 5.** μ could enter in two places — fitting
and injection — and by default it enters **only at injection**
(`--mc_fit_centred false`, the default). So `C` is fitted on raw `X`, identical
to method 5's, and the two methods differ by the bias-recovery term alone. That
is the clean one-variable comparison, and it matches the derivation, which
starts from the standard `h' = Ch`.

`--mc_fit_centred true` additionally fits `C` on `(X − μ)`. That changes two
things at once — the matrix *and* the injection rule — so a difference against
method 5 can no longer be attributed to the bias term. Those runs are tagged
`_conceptor_mc_fitmc.csv` to keep them separate. Use it as a deliberate
ablation, not as the main comparison.

> Note the aperture consequence: with the default, `C` has the same spectrum as
> method 5's, so `--list_apertures_mean_centered` should be set to the **same
> values** as `--list_apertures_normal`. The separate flag exists for the
> `true` case, where centred activations shift the useful aperture range.

**Setting `β = 1` makes the operation purely contractive.** Since `C`'s
eigenvalues are < 1, `βC − I` is then negative semi-definite and the update can
only pull `h` toward `μ`. `β > 1` is what allows amplification along task
directions.

---

## The corpus mean `μ`

Needed by `addition-mean` and `conceptor-mean` only. It is **not** derived from
task data — it is the model's average residual over ~600 files of generic news
prose, computed once by
`Mean_Activation_Vectors/Mean_train/mean_activation.py`:

```bash
cd ~/meanact && python3 mean_activation.py     # -> mean_activations_qwen1.5b.pkl
cp mean_activations_qwen1.5b.pkl <project>/Results/
```

A pickle of `{layer_index: tensor(d_model)}`. One file serves every task; it
only needs regenerating if you change the model.

Note it covers layers **1..n_layers−1** — layer 0 is absent, so do not include
0 in `--list_extraction_layers` for the mean-based methods.

---

## Method summary

| `--experiment_type` | Artifact | Injection | Knobs | Needs μ | Output suffix |
|---|---|---|---|---|---|
| `baseline` | — | none | — | no | `_baseline` |
| `addition` | `v = mean(X)` | `h + βv` | β | no | `_addition` |
| `addition-mean` | `v = mean(X)` | `h + β(v − μ)` | β | **yes** | `_addition_mc` |
| `conceptor-paper` | `C = R(R+α⁻²I)⁻¹` | `βCh` | β, α | no | `_conceptor_paper` |
| `conceptor` | same `C` | `h + λ(βC − I)h` | β, α, λ | no | `_conceptor` |
| `conceptor-mean` | same `C` | `h + λ((βC − I)h + b)` | β, α, λ | **yes** | `_conceptor_mc` |
| `conceptor-mean --mc_fit_centred true` | `C` on centred `X` | same as above | β, α, λ | **yes** | `_conceptor_mc_fitmc` |

Files are named `Qwen2.5-1.5B-Instruct_<task>_<suffix>.csv`. `_get_filename`
appends `_1`, `_2`, … rather than overwriting an existing file.

**All three conceptor variants share one `C` by default.** That makes the
comparison a ladder where each step adds exactly one thing:

```
conceptor-paper   h' = βCh                            the published form
conceptor         h' = h + λ(βC − I)h                 + λ blend
conceptor-mean    h' = h + λ((βC − I)h + b)           + bias recovery
```

so any accuracy difference is attributable to the term that was added, not to a
differently-fitted matrix. `--mc_fit_centred true` breaks that property on
purpose and is a separate ablation.

---

## Which flag feeds which method

Every `--list_*` flag is parsed for every run, but each method reads only the
subset it needs and **silently ignores the rest**. A flag that looks accepted
may do nothing for the method you care about.

| Flag | baseline | addition | addition-mean | conceptor-paper | conceptor | conceptor-mean |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `--list_extraction_layers` | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `--list_beta_averaging` | — | ✓ | ✓ | — | — | — |
| `--list_beta_conceptor` | — | — | — | ✓ | ✓ | ✓ |
| `--list_apertures_normal` | — | — | — | ✓ | ✓ | — |
| `--list_apertures_mean_centered` | — | — | — | — | — | ✓ |
| `--list_lambda_conceptor` | — | — | — | pinned 1.0 | ✓ | ✓ |
| `--mean_activations_path` | — | — | ✓ | — | — | ✓ |
| `--mc_fit_centred` | — | — | — | — | — | ✓ |

Shared by everything: `--task`, `--num_steering_prompts` (extraction set size),
`--num_input_prompts` (evaluation set size), `--num_experiments`,
`--batch_size`, `--seed`, `--results_path`, `--model_name`, `--dtype`.

Three traps worth knowing:

- **`conceptor-mean` reads a different aperture flag.** It uses
  `--list_apertures_mean_centered` while the other two read
  `--list_apertures_normal`. With the default `--mc_fit_centred false` the
  conceptors are identical across all three, so **set both flags to the same
  values** — otherwise `conceptor-mean` sweeps a different grid and the ladder
  comparison is unfair. (The separate flag exists for `--mc_fit_centred true`,
  where centred activations shift the useful aperture range.) Leave one unset
  and it silently falls back to its argparse default of `[0.05]`.
- **`conceptor-paper` ignores λ.** `h' = βCh` has no λ term, so
  `run_experiment.py` pins it to 1.0 and prints a note. Sweeping four λ values
  would otherwise run the identical experiment four times.
- **`baseline` has no layer dimension.** It ignores every `--list_*` flag and
  produces a single accuracy figure, not a per-layer curve.

### Grid size per method

The number of configurations is the product of the lists that method reads.
With 12 layers, 7 β_add, 4 β_conc, 5 apertures (**both** lists set to the same
5 values, per the trap above), 4 λ, 1000 eval prompts and 1 experiment:

| Method | Grid | Configs | Forward passes |
|---|---|---:|---:|
| `addition` | L × β_add | 84 | 84,000 |
| `conceptor-paper` | L × β_conc × ap | 240 | 240,000 |
| `conceptor` | L × β_conc × ap × λ | 960 | 960,000 |
| `conceptor-mean` | L × β_conc × ap × λ | 960 | 960,000 |
| **total** | | **2,244** | **2,244,000** |

Forward passes = configs × `--num_experiments` × `--num_input_prompts`. Note
these are *prompts*, not GPU calls: at `--batch_size 128` the total above is
~17,500 actual forward passes, and each is a single pass over ~5 tokens per
prompt — no generation. That is why a sweep this size finishes in minutes.

The conceptor variants dominate: addition is under 4% of the work, and λ alone
quadruples both residual conceptor grids. `conceptor-paper` is cheap precisely
because it has no λ dimension — adding it to a run that already includes the
other three costs about 12% more.

`conceptor` and `conceptor-mean` now have **equal** grids, which is the point:
equal grids over an identical `C` make their difference attributable to the bias
term alone.

`--num_experiments 1` means the `Average` row is a copy of the single run, with
no variance estimate. Raising it multiplies the whole cost linearly.

---

## Running

```bash
cd src/functionvectors

python3 run_experiment.py \
    --experiment_type addition,conceptor-paper,conceptor,conceptor-mean \
    --task capitalize \
    --model_name Qwen/Qwen2.5-1.5B-Instruct --dtype float16 \
    --mean_activations_path ../../Results/mean_activations_qwen1.5b.pkl \
    --list_extraction_layers 13,14,15,16,17,18,19,20,21,22,23,24 \
    --list_beta_averaging 0.5,1.0,2.0,2.3,3.0,4.0,5.0 \
    --list_beta_conceptor 1.0,2.0,3.9,5.0 \
    --list_apertures_normal 0.005,0.05,0.1,1.0,2.0 \
    --list_apertures_mean_centered 0.005,0.05,0.1,1.0,2.0 \
    --list_lambda_conceptor 0.25,0.5,0.75,1.0 \
    --num_steering_prompts 100 --num_input_prompts 1000 \
    --num_experiments 1 --batch_size 128 \
    --seed 0 --results_path ../../Results_v2
```

Both aperture lists carry the same values — required for a fair ladder, since
all three conceptor variants share one `C` by default.

To additionally run the two-changes-at-once ablation:

```bash
python3 run_experiment.py --experiment_type conceptor-mean --mc_fit_centred true ...
# -> Qwen2.5-1.5B-Instruct_<task>_conceptor_mc_fitmc.csv
```

Or on SLURM: `sbatch run_all.sh` from the project root (create `logs/` first).

Results checkpoint to CSV every 5 configs, so an interrupted sweep leaves
usable partial output. Layers are the **outermost** loop, so a partial file
contains complete low layers rather than a thin slice of all of them.

## Plotting

```bash
cd Results
python3 plot.py --task capitalize          # defaults to ../Results_v2
python3 plot.py --task antonyms
```

One line per method, each point being that method's best configuration at that
layer. Writes a PNG plus a companion CSV listing every point and its winning
hyperparameters.

## See also

`EXPERIMENT_SCRIPTS.md` — what each script file does, how `run_experiment.py`
and `run_merged_experiment.py` differ, and known issues in the merged runner.
