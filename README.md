# Conceptor Steering for Open-Ended Emotional Generation

**Can covariance-aware *conceptor* steering — which beats additive steering vectors on
structured tasks — control the *emotion* of free-form LLM responses?**

This repository is a systematic study of activation steering on
**Qwen2.5-1.5B-Instruct** (28 layers, d = 1536). It compares additive emotion vectors
with conceptor-based steering, diagnoses why off-the-shelf conceptors break down on
open-ended generation, and develops and tests a fix: a mean-shift + conceptor rule
fitted on emotion-specific minimal-pair data.

> **Status:** active research. The additive-vector and structured-task results are
> complete. The conceptor failure diagnosis is confirmed; the fixed method is being
> swept on the IIIT Ada cluster (2 emotions first, then all 5). Numbers below are
> marked as final or preliminary.

Team: Paarth Jindal, Virat Garg, Varun Gupta (ANLP project).
Write-ups: [`research.pdf`](research.pdf) (report),
[`research_proofs.pdf`](research_proofs.pdf) (earlier report with conceptor proofs
appendix), [`papers/anlp_proofs.pdf`](papers/anlp_proofs.pdf) (derivations for the
method ladder below).

---

## Why this is interesting

Most steering work evaluates on tasks with a single right token: antonyms,
capitalisation, translation. There, a conceptor — a soft projection matrix
`C = R(R + α⁻²I)⁻¹` built from activation covariance — is known to outperform a single
steering vector. Emotion is different. It is a *style* spread over a whole free-form
response, the activations carry strong mean shifts, and success has to be judged on
fluent text, not one token. This project asks whether the conceptor advantage carries
over, and when it does not, why.

## Key findings

| # | Finding | Status |
|---|---|---|
| 1 | **Conceptors beat addition on structured tasks.** Zero-shot capitalisation: residual conceptor **55.0%** vs addition 39.6% (unsteered 0%). Antonyms: conceptor **29.0%** vs addition 11.0%. | final |
| 2 | **Additive emotion vectors steer emotion while keeping text coherent.** BART-MNLI target confidence 0.12 → 0.37 (α=2) → 0.56 (α=4); joy reaches 0.94. LLM-judged emotion strength (EAS) rises 0.30 → 1.18 at α=2. Degenerate outputs: 0%. | final |
| 3 | **Emotion becomes linearly separable late.** Silhouette score of per-query emotion vectors rises from about 0.25–0.33 (layers 0–15) to **0.57 at layer 27**. | final |
| 4 | **Naïve conceptor emotion steering collapses into word salad.** The earlier run reported "86% improved", but **100%** of its steered outputs were degenerate ("OhWowAhOhImagine…"). The classifier rewarded noise. | final |
| 5 | **Three separate causes, each confirmed by direct measurement** (see [Diagnosis](#diagnosis-why-conceptor-emotion-steering-failed)): norm explosion, statistics fitted at the wrong token position, and low-rank C erasing token content. | final |
| 6 | **Fix: shift + conceptor on mid layers, fitted on generated / teacher-forced tokens**, gives readable emotional text where every earlier conceptor variant gave noise. | preliminary, sweep running |

Figures: [`plots/`](plots) (emotion results) and
[`conceptor_antonym/capitalise_final5/`](conceptor_antonym/capitalise_final5),
[`conceptor_antonym/capitalise_final6/Results_v3/`](conceptor_antonym/capitalise_final6/Results_v3)
(structured tasks). PNGs are gitignored; regenerate them with the plotting scripts.

---

## Methods

Notation: `h` is the residual-stream activation at layer ℓ for the current token, `μ`
is a mean activation, `C` a conceptor, `β`/`α`/`λ` are strength, aperture and
residual blend.

| Method | Update | Idea |
|---|---|---|
| Additive (ActAdd / emotion vector) | `h + β·v`, `v = μ_emotion − μ_neutral` | the least-squares best single translation from neutral to emotional activations |
| Conceptor (published) | `β·C·h` | covariance-aware soft projection onto the concept's subspace |
| Residual conceptor | `h + λ(βC − I)h` | interpolates between the base model (λ=0) and the published form (λ=1) |
| Mean-corrected residual (CoRAL) | `h + λ(βC − I)(h − μ)` | preserves the activation centroid |
| **Shift + conceptor (this work)** | `h + λ[(μₑ − μₙ) + (Cₑ − I)(h − μₙ)]` | additive mean shift plus a conceptor reshaping of the deviation from the neutral mean. With C = I it reduces to the plain mean shift |

Derivations — the mean difference as a least-squares optimum, the conceptor as a
regularised reconstruction operator, bounds on the residual intervention, and mean
preservation — are in [`papers/anlp_proofs.pdf`](papers/anlp_proofs.pdf).
Δ-conceptor and discriminative-conceptor theory is in the appendix of
[`research_proofs.pdf`](research_proofs.pdf).

### Diagnosis: why conceptor emotion steering failed

Measured on the cached Qwen model, with greedy decoding on held-out queries:

1. **Expansive update compounded over depth.** `h += β·C(h − μ)` has eigenvalues up to
   1 + β. Applied at all 28 layers, the residual norm at layer 27 goes from **173 to
   ~1.1×10⁷**. This rule also pushes `h` *away* from the emotion mean, not toward it.
2. **Statistics from the wrong token position.** `μ` and `C` came from the prompt's
   final token (the "about to start a reply" state) but were applied to every
   generated token. With the norm under control, every token is still pulled toward
   "start a reply", which produces reply openers glued without spaces: `OhIOhOh…`,
   `AsAsAsWhenIt…`. Additive vectors avoid this because `μₑ − μₙ` cancels the
   position effect.
3. **Low-rank C erases content.** About 100 samples in 1536 dimensions gives
   `C − I ≈ −I` almost everywhere, so a residual update replaces `h` with `μ`:
   `the the the…`.

**Fix:** steer a few mid layers; fit `μ`/`C` on *response* tokens (generated or
teacher-forced); include the mean shift; use a small λ; fit on many tokens.

### Data for fitting

| Source | What | Why |
|---|---|---|
| `model_gen` | the model's own generated tokens under a "respond with strong *X*" system prompt | matches the steering-time distribution |
| `synthetic` | **250 minimal-pair sets from GPT-5.5**: for each query, one neutral reply and five emotional rewrites with *identical facts*. Teacher-forced under the neutral prompt | the emotional minus neutral difference isolates emotion from content and prompt artifacts |
| `goemotions` | 400 human Reddit comments per class with a single GoEmotions label (joy, anger, disgust, fear, sadness, neutral) | real human emotional text |

Example minimal pair (`conceptor_emotion/data/emotion_pairs_synthetic.jsonl`):

> **neutral** — I would wait one day, then remind them that my birthday was yesterday…
> **anger** — I would wait one *boiling* day, then *confront* them… and *demand* they plan a coffee…
> **sadness** — I would wait one *heavy* day, then *quietly* remind them… *like a stone in my chest*…

### Evaluation

- **Target-emotion confidence**: zero-shot BART-large-MNLI over
  {joy, anger, disgust, fear, sadness, neutral}.
- **Degenerate-output rate**: under 5% spaces (glued word salad), or under 40% distinct
  words (repetition). The sweep scores degenerate outputs as 0 ("valid confidence").
- **Perplexity** of the response under the unsteered model, given the prompt.
- **EAS** (Emotion Absolute Strength): an LLM judge scores six emotions 0–100;
  `EAS = Σ (score/100)²`.
- Structured tasks: zero-shot exact match of the first predicted token.

---

## Directory structure

```
qwen_testing/
├── README.md
├── research.pdf                  # project report
├── research_proofs.pdf           # earlier report + conceptor proofs appendix
├── env_config.py                 # .env loader used by every script (no dependencies)
├── .env.example                  # template for API keys (copy to .env, which is gitignored)
│
├── papers/                       # references + our derivations
│   ├── 2410.16314v4.pdf          #   Postmus & Abreu — Steering LLMs using Conceptors
│   ├── 2502.04075v1 (1).pdf      #   Dong et al. — Controllable Emotion Generation with Emotion Vectors
│   ├── 2605.04980v1.pdf          #   Triantafyllopoulos et al. — Conceptors for Semantic Steering
│   └── anlp_proofs.pdf           #   addition → conceptor → residual → mean-corrected derivations
│
├── extraction/                   # emotion vectors + representation geometry
│   ├── qwen_emotion_extraction.py#   emotional − neutral hidden states, all 29 layers (averaged / per-prompt)
│   ├── tsne_emotion_vectors.py   #   per-layer t-SNE of per-prompt vectors
│   ├── silhouette_by_layer.py    #   cluster separability per layer
│   ├── silhouette_plot.py
│   ├── emotion_vectors/          #   Qwen25_1B5_<emotion>_vectors.json (+ base, summary)
│   └── tsne_plots/               #   layer_XX.png / .csv
│
├── confidence_score/             # additive EV steering scored by BART-MNLI
│   ├── confidence_score_evaluator.py   # all-layer or layer-wise (--layerwise) injection
│   ├── LAYERWISE_README.md
│   └── alpha=-1.0/2.0/4.0.csv|json     # results per steering strength
│
├── EAS_score/                    # additive EV steering scored by an LLM judge (EAS)
│   ├── eas_score_evaluator.py
│   ├── EAS_README.md
│   └── EAS_alpha=-1.csv, EAS_alpha=2.0.csv
│
├── EAS_validation/               # consistency checks of the LLM judge on GoEmotions
│   ├── gpt4o_eas_calculator.py, gpt4o_consistency_test.py
│   └── dataset_new/              #   GoEmotions splits (gitignored; download separately)
│
├── conceptor_antonym/            # structured-task test bed (function-vector style, zero-shot)
│   ├── README.md                 #   the six methods, flags, grid sizes — read this first
│   ├── EXPERIMENT_SCRIPTS.md
│   ├── run_all.sh                #   SLURM launcher
│   ├── src/functionvectors/      #   run_experiment.py + utils/{conceptors,steering,actadd,data}.py
│   ├── data/functionvectors/     #   antonyms, capitalize, translation, … task pairs
│   ├── Results/                  #   early runs + plot.py / plot_lambda.py
│   ├── capitalise_final5/        #   final capitalize sweep + layer×{β,λ,α} heatmaps
│   └── capitalise_final6/        #   final antonym sweep (Results_v3/)
│
├── capitalise/                   # earlier standalone capitalize experiments (conceptor + additive)
│
├── conceptor_emotion/            # conceptor steering for open-ended emotion  ← current focus
│   ├── conceptor_extraction.py   #   original: prompt-end-token conceptors (shown to fail)
│   ├── conceptor_emotion_evaluation.py  # original evaluator (h += βC(h−μ), all layers)
│   ├── conceptor_evaluation_beta4.0_aperture0.2_*.csv|json   # the degenerate run, kept for the record
│   ├── build_emotion_datasets.py #   NEW: GPT-5.5 minimal pairs + GoEmotions single-label sets
│   ├── shift_conceptor_sweep.py  #   NEW: fix sweep — sources × layers × λ × aperture vs project EVs
│   ├── run_sweep.sh              #   NEW: SLURM job for Ada
│   ├── data/
│   │   ├── emotion_pairs_synthetic.jsonl    # 250 × {query, neutral, joy, anger, disgust, fear, sadness}
│   │   └── goemotions_emotion_texts.json    # 400 texts per class
│   └── EmotionQuery.json, EmotionQuery+.json  # queries (fit: [0,250), test: [300,…))
│
├── plots/
│   ├── plot_emotion_results.py   #   conceptor vs baseline, EV confidence vs α, EAS
│   └── plot_sweep.py             #   best config per method, tradeoff scatter, λ×aperture heatmaps
│
└── local_qwen_markings/          # exploratory Kaggle notebook
```

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch transformers pandas numpy matplotlib tqdm scikit-learn requests
cp .env.example .env      # then fill in the keys you need
```

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `conceptor_emotion/build_emotion_datasets.py` (synthetic pairs) |
| `AZURE_OPENAI_*` | EAS judges (`EAS_score/`, `EAS_validation/`) |
| `HF_TOKEN` | only for gated models (not needed for Qwen2.5 or BART-MNLI) |

## Reproducing

**Emotion vectors and geometry**
```bash
cd extraction
python3 qwen_emotion_extraction.py --mode both --num_queries 100 --output_dir ./emotion_vectors
python3 tsne_emotion_vectors.py && python3 silhouette_by_layer.py
```

**Additive EV steering**
```bash
cd confidence_score && python3 confidence_score_evaluator.py --alpha 2.0 --num_queries 50
cd EAS_score && python3 eas_score_evaluator.py --alpha 2.0 --num_queries 100
```

**Structured tasks (conceptor vs addition)** — see `conceptor_antonym/README.md`
```bash
cd conceptor_antonym/src/functionvectors
python3 run_experiment.py --experiment_type addition,conceptor-paper,conceptor,conceptor-mean \
    --task capitalize --model_name Qwen/Qwen2.5-1.5B-Instruct ...
```

**Conceptor emotion steering — the fix**
```bash
cd conceptor_emotion
python3 build_emotion_datasets.py --source goemotions
python3 build_emotion_datasets.py --source synthetic --n_queries 250 --model gpt-5.5
python3 shift_conceptor_sweep.py --quick                 # 3-minute end-to-end smoke test
python3 shift_conceptor_sweep.py --emotions joy,sadness  # sweep; writes sweep_results/
python3 ../plots/plot_sweep.py sweep_results
```

**On the Ada cluster (SLURM)** — request the GPU through `sbatch`:
```bash
cd ~/qwen_testing/conceptor_emotion
sbatch -A cvit -p u22 run_sweep.sh                                   # joy, sadness
EMOTIONS=joy,anger,disgust,fear,sadness EMOTIONS_TAG=5emo sbatch -A cvit -p u22 run_sweep.sh
```
`run_sweep.sh` activates `~/venv`, sets `HF_HUB_OFFLINE=1` (models pre-cached), and
checks CUDA before loading anything. Add `--exclude=<node>` if a node reports
`CUDA unknown error`.

**Plots of the existing emotion results**
```bash
python3 plots/plot_emotion_results.py
```

---

## Known issues and caveats

- **Apple MPS:** SDPA attention returns NaN logits for left-padded batch rows in every
  dtype. The sweep uses eager attention on MPS automatically.
- **Project EV injection** (`confidence_score_evaluator.py`) hooks every module named
  `layers.N…` with at most 3 dots, so the vector is added at the layer output *and* both
  layernorms (three times per layer). Vector `layer_N`, whose index 0 is the embedding,
  is injected at decoder layer N, an off-by-one. `shift_conceptor_sweep.py` reproduces
  this exactly for a fair "project EV" baseline.
- The original `conceptor_emotion` evaluation fitted and tested on the same first 100
  queries. The new sweep fits on `[0, 250)` and tests on `[300, …)`.
- The classifier is gameable: always report the degenerate-output rate next to
  confidence.
- The EAS judge in code is GPT-4o-mini (Azure); the report says Gemini-2.5-Flash.
- Single model (Qwen2.5-1.5B). Cross-architecture replication is future work.

## Roadmap

- [x] Additive EVs: layer, strength and geometry analysis
- [x] Conceptor ≫ addition on structured tasks (capitalize, antonyms)
- [x] Diagnose conceptor collapse on open-ended emotion
- [x] Emotion-specific fitting data: GPT-5.5 minimal pairs, GoEmotions
- [ ] Shift+conceptor sweep, 2 emotions → 5 emotions (running on Ada)
- [ ] Human evaluation of steered text; LLM-judge EAS on the best configs
- [ ] Δ-conceptor from paired differences; Boolean composition (e.g. anger ∧ sadness)
- [ ] Cross-model replication (Llama-3, Mistral)

## References

- Postmus & Abreu (2024). *Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering.* arXiv:2410.16314
- Dong et al. (2025). *Controllable Emotion Generation with Emotion Vectors.* arXiv:2502.04075
- Triantafyllopoulos et al. (2026). *Conceptors for Semantic Steering.* arXiv:2605.04980
- Jaeger (2014). *Controlling Recurrent Neural Networks by Conceptors.* arXiv:1403.3369
- Jaeger (2014). *Conceptors: an easy introduction.* arXiv:1406.2671
- Turner et al. (2023). *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248
- Todd et al. (2023). *Function Vectors in Large Language Models.* arXiv:2310.15213
- Jorgensen et al. (2023). *Improving Activation Steering in Language Models with Mean-Centring.* arXiv:2312.03813
- Rimsky et al. (2023). *Steering Llama 2 via Contrastive Activation Addition.* arXiv:2312.06681
- Demszky et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* arXiv:2005.00547
