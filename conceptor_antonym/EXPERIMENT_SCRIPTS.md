# Experiment scripts — roles and differences

Covers the three files in `src/functionvectors/`:

| File | Lines | Kind | Role |
|---|---|---|---|
| `experiments.py` | 400 | **library** | The evaluation loops. Never run directly. |
| `run_experiment.py` | 218 | **entry point** | Single-task steering experiment. |
| `run_merged_experiment.py` | 275 | **entry point** | Two-task composition experiment. |

The two `run_*` scripts are alternative drivers over the *same* library. Neither
imports the other; both import the same three functions from `experiments.py`.

```
run_experiment.py ─┐
                   ├─→ experiments.py ─→ utils/steering.py (hooks, forward passes)
run_merged_experiment.py ─┘
```

---

## `experiments.py` — the evaluation library

Three public functions, all sharing the same measurement: build a prompt batch,
optionally install a steering hook on `blocks.{layer}.hook_resid_pre`, take the
model's **top-1 first generated token**, and count it correct if it matches the
expected output. Accuracy = exact-match rate on that single token.

| Function | Steering applied | Config swept |
|---|---|---|
| `run_baseline_experiments` | none (`fwd_hooks=[]`) | — |
| `run_all_conceptor_experiments` | `h' = h + λ(βC − I)h` | layer × beta × aperture × lambda |
| `run_all_actadd_experiments` | `h' = h + β·v` | layer × beta |

Supporting details:

- `run_conceptor_experiment` is the inner single-config loop; the `run_all_*`
  functions wrap it in a `tqdm` sweep over configs.
- Both `run_all_*` functions **checkpoint to CSV every 5 configs** (and on the
  last one), so a long sweep that dies partway still leaves partial results.
- `_get_filename` auto-suffixes `_1`, `_2`, … so a re-run never silently
  overwrites an earlier CSV. That is why `Results/` holds
  `..._conceptor_1.csv`, `..._addition_1.csv`, and so on.
- Passing `mean_train_activations` switches both methods to their mean-centred
  hook variant and adds `_mc` to the output filename.
- Each function appends an `experiment: "Average"` row after the per-experiment
  rows.

---

## `run_experiment.py` — single-task driver

One task, one steering method, one sweep. Dispatches on `--experiment_type`
(`ExperimentType` in `utils/configs.py`):

| `--experiment_type` | What runs |
|---|---|
| `baseline` | unsteered accuracy |
| `conceptor` | conceptor steering, `--list_apertures_normal` |
| `conceptor-mean` | mean-centred conceptor, `--list_apertures_mean_centered` |
| `addition` | ActAdd steering vector |
| `addition-mean` | mean-centred ActAdd |

Pipeline: parse args → load mean activations (mean variants only) → load
`HookedTransformer` → `generate_experiment_data` → `generate_activations` →
build the config grid → cache conceptors / activation vectors → run the sweep.

Data comes from a single file, `../../data/functionvectors/{task}.json`.
See `commands.txt` for a working invocation.

---

## `run_merged_experiment.py` — two-task composition driver

Answers a different question: **if you have a steering artifact for task A and
one for task B, can you combine them to steer a task that requires both?**

`--task` must contain an ampersand (asserted at line 28), e.g.
`english-french&antonyms`. That one flag is expanded into **three** datasets:

| Key | File |
|---|---|
| `first` | `english-french.json` |
| `second` | `antonyms.json` |
| `combined` | `english-french-antonyms.json` |

Every stage — experiment data, activations, conceptor cache — is built three
times, once per key. Evaluation always happens on the `combined` prompts; only
the *steering artifact* varies. That is the comparison the script exists to make:

**Conceptor branch** — three runs, three CSV suffixes:

| Suffix | Steering artifact | Question |
|---|---|---|
| `_boolean` | `combine_conceptors(C₁, C₂)` — soft **OR** | does composing separately-learned conceptors work? |
| `_booleanand` | `combine_conceptors_and(C₁, C₂)` — soft **AND** | does AND work as well as OR? |
| `_combined` | conceptor fit directly on combined data | the reference ceiling |

**Addition branch** — two runs:

| Suffix | Steering artifact |
|---|---|
| `_combined` | vector fit on combined data |
| `_merged` | `(v₁ + v₂) / 2` |

The Boolean operators live in `utils/conceptors.py`; its own docstrings note
that OR "works much better than AND".

---

## Differences at a glance

|  | `run_experiment.py` | `run_merged_experiment.py` |
|---|---|---|
| Tasks per run | 1 | 2 + their combination |
| `--task` format | `capitalize` | `taskA&taskB` (assert) |
| Datasets loaded | 1 | 3 |
| Runs per invocation | 1 sweep | 3 sweeps (conceptor) / 2 (addition) |
| Conceptor combination | — | OR, AND, and direct-fit reference |
| `conceptor-mean` | supported | `NotImplementedError` |
| `addition-mean` | supported | `NotImplementedError` |
| `--batch_size` | honoured | **ignored** — hardcoded 50 / 150 |
| `--list_lambda_conceptor` | swept | **ignored** — `lambda_` defaults to 0.5 |
| `TOKENIZERS_PARALLELISM` | set | not set |
| `hf_token.txt` auto-load | yes | no |
| CSV `fname_suffix` | not used | used to separate the runs |

---

## Known issues

Three problems in `run_merged_experiment.py`, all verified against the code.
`run_experiment.py` is unaffected by all three.

### 1. The addition branch crashes immediately

Lines 237 and 260 call `run_all_actadd_experiments(..., fname_suffix=...)`, but
that function does not accept a `fname_suffix` parameter — only
`run_all_conceptor_experiments` does. Any run with `--experiment_type addition`
raises:

```
TypeError: run_all_actadd_experiments() got an unexpected keyword argument 'fname_suffix'
```

Fix: add `fname_suffix: str = ""` to `run_all_actadd_experiments` in
`experiments.py` and thread it into its `_get_filename` call, matching how
`run_all_conceptor_experiments` already does it.

### 2. The AND result is computed from the OR result

Line 146 does `conceptors_cache["boolean"] = conceptors_cache["first"]`. That
binds the same dict, not a copy, so the OR loop at 147–152 **overwrites
`conceptors_cache["first"]` in place**. When line 170 re-aliases `"first"` for
the AND pass, `"first"` no longer holds C₁ — it holds the OR result. So the run
labelled `_booleanand` actually evaluates:

```
AND(OR(C₁, C₂), C₂)        instead of        AND(C₁, C₂)
```

Reproduced with the exact pattern:

```
boolean is first?  True
after OR pass   -> first[0]['k']   = OR(C1,C2)
after AND pass  -> boolean[0]['k'] = AND(OR(C1,C2),C2)
```

This invalidates any `_booleanand` CSV, and it means the AND-vs-OR comparison —
the point of the branch — is not measuring what it claims. Fix with
`copy.deepcopy(conceptors_cache["first"])`, or build a fresh dict per operator.

Line 252 has the same aliasing (`"jointactivations" = "first"`), but `"first"`
is not read again afterwards, so it is currently harmless — still worth fixing
so it does not become a bug when the branch is extended.

### 3. Two swept parameters are silently ignored

`--batch_size` and `--list_lambda_conceptor` are parsed and printed in the
argument banner, but the merged script hardcodes `batch_size=50`/`150` and omits
`lambda_` when building `ExperimentConfigConceptor`, so it falls back to the
`0.5` default in `utils/configs.py`. A lambda sweep passed on the command line
looks accepted and changes nothing.

---

## Reference — the underlying maths

Conceptor steering, `utils/steering.py:steer`:

```
h' = h + λ (β C − I) h
```

`λ = 0` disables steering, `λ = 1` applies the full update. `C` comes from
`compute_conceptor(X, aperture)`; the hook is installed on
`blocks.{layer}.hook_resid_pre`.

ActAdd steering, `generate_ave_hook_addition`:

```
h' = h + β v        where v is the mean task activation at that layer
```

Boolean composition, `utils/conceptors.py`:

```
OR (combine_conceptors)      : I − ((I−C₁)⁻¹ + (I−C₂)⁻¹ − I)⁻¹
AND (combine_conceptors_and) : (C₁⁻¹ + C₂⁻¹ − I)⁻¹
```
