# Open-ended emotion steering: methods, evaluation protocol and prompts

This document describes every steering method compared for **open-ended emotional generation**
on **Qwen2.5-1.5B-Instruct** (28 decoder layers, hidden size d = 1536), how each is fitted and
applied, how configurations are selected and tested, and the exact prompts used — including the
LLM-judge prompt used for the human-like evaluation.

Code: `shift_conceptor_sweep.py` (all methods + sweeps), `benchmark_test.py` (held-out benchmark),
`llm_judge.py` (LLM-judge scoring), `build_emotion_datasets.py` (fitting data).

---

## 1. Common setup

| Item | Setting |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16 (CUDA) / bf16 + eager attention (Apple MPS) |
| Target emotions | joy, anger, disgust, fear, sadness (each steered and evaluated separately) |
| Generation prompt | Neutral system prompt (below) + user query, Qwen chat template. Emotion must come from the steering, not the prompt. |
| Decoding | Greedy, up to 40 new tokens, one reply per prompt |
| Where steering acts | Forward hook on a decoder layer's **output** (residual stream after the block), at the **current token** at every step: the last prompt token during prefill, then each generated token (exception: Interpolation steers all tokens, §3.6) |
| Queries | `EmotionQuery+.json` (400 everyday user messages) |

**Neutral system prompt (used for every generation and for fitting):**
> You are an AI assistant. Respond neutrally and objectively to the user's query without showing any particular emotion. Maintain a balanced and factual tone.

### Prompt splits (no overlap between roles)

| Prompts | Role |
|---|---|
| EmotionQuery+ [0, 250) | Fitting data: the gpt-5.5 minimal pairs are written for these queries |
| EmotionQuery+ [300, 350) | **Selection set** — every configuration of every sweep is generated and scored here (50 prompts/emotion) |
| EmotionQuery+ [250, 300) ∪ [350, 400) | **Held-out test set** — only the one selected configuration per method is run here (100 prompts/emotion) |

---

## 2. Fitting data (statistics μ and C)

### 2.1 gpt-5.5 minimal pairs (`data/emotion_pairs_synthetic.jsonl`, 250 queries)

For each query, gpt-5.5 writes six replies with **identical facts** that differ **only in tone**.
Generation prompt (`build_emotion_datasets.py`, `reasoning_effort=low`, JSON mode):

```
A user asked an assistant: "{query}"

Write six replies that convey the SAME content (same points, same facts, similar length of 2-3 sentences, first person) and differ ONLY in emotional tone:
- neutral: calm, factual, flat; no emotional words at all
- joy, anger, disgust, fear, sadness: that emotion is INTENSE and saturates every sentence through vivid word choice, imagery and rhythm (exclamations, interjections where natural), so a reader would identify it instantly
Keep the facts identical across all six. Avoid naming the emotion itself (e.g. don't write "I feel sad"). Return JSON with keys neutral, joy, anger, disgust, fear, sadness.
```

### 2.2 How activations become the matrix X (teacher forcing)

Each reply is inserted as the assistant's turn after the neutral system prompt and its query, and
run through Qwen in one forward pass. The hidden state of layer ℓ at **every reply token** becomes
one row of X (prompt tokens are dropped). Rows are stacked over all 250 replies:

- `X_e` (emotion e): ≈ 14.5k–15.4k token rows × 1536 per layer
- `X_n` (neutral): ≈ 11–12k token rows × 1536 per layer

Derived statistics, per layer: `μ_e = mean(X_e)`, `μ_n = mean(X_n)`,
covariance `Σ_e = cov(X_e)` (mean-centred), correlation `R_e = X_eᵀX_e / n` (uncentred).

A conceptor with aperture α is `C(M, α) = M (M + α⁻² I)⁻¹` (computed via eigendecomposition),
a soft projection whose eigenvalues λᵢ/(λᵢ + α⁻²) lie in [0, 1).

---

## 3. Methods for open-ended generation

`h` is the hidden state being steered. "Stacked" = applied at every layer of a contiguous block.

### 3.1 Baseline
No steering; same neutral prompt and decoding.

### 3.2 Project EV (prior emotion vectors; Dong et al. 2025 style)
- **Vector:** for 100 queries, Qwen *generates* a reply under the system prompt
  "Respond with strong {e} emotion. Express clear {e} in your tone, word choice, and perspective."
  and under the neutral prompt; hidden states are mean-pooled over the prompt (as one step) and the
  generated tokens; `v = mean_q(h̄_emotional(q) − h̄_neutral(q))` per layer (`extraction/`).
- **Update:** `h' = h + α · v/‖v‖` at **all 28 layers**, replicating
  `confidence_score/confidence_score_evaluator.py` exactly (its hook also fires on both layernorms
  of each layer, so the vector is added three times per layer, with an off-by-one layer index).
- **Grid:** α ∈ {1, 2, 3, 4, 5}.

### 3.3 Plain shift (additive; ablation of ours)
- **Update:** `h' = h + λ (μ_e − μ_n)`, stacked over a layer block.
- Same data and layers as ours; differs only by the missing conceptor term.
- **Grid:** 10 layer blocks {4-10, 6-12, 8-14, 10-16, 12-18, 14-20, 16-22, 18-24, 8-20, 10-14} × λ ∈ {0.5, 1, 1.5, 2, 2.5}.

### 3.4 Shift + conceptor — **ours** (stacked)
- **Update:** `h' = h + λ [ (μ_e − μ_n) + (C_e − I)(h − μ_n) ]`, with `C_e = C(Σ_e, α)`, stacked over a layer block.
  - first term: additive mean shift toward the emotion;
  - second term: conceptor correction — keeps the components of the deviation from the neutral
    mean that lie in the emotion's activation subspace and shrinks the rest.
- **Property:** the linear part `I + λ(C_e − I)` has eigenvalues in [1 − λ, 1] for λ ≤ 1, so the
  update never enlarges the hidden state. This is what allows stacking over many layers without
  the norm blow-up of multiplicative rules; too large λ or too many layers instead over-shrink
  (erase token content).
- **Grid:** the same 10 layer blocks × α ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0} × λ ∈ {0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5} (560 configs/emotion).

### 3.5 Published conceptor, replace form — βCh (Postmus & Abreu, arXiv:2410.16314)
- **Update:** `h' = β · C_e h`, with `C_e = C(R_e, α)` (uncentred correlation, as in the paper).
- Single layer at a time (the published setting). The original paper evaluates only next-token
  function tasks; this is its rule applied to open-ended generation.
- **Grid (benchmark selection pool):** every layer 0–27 individually. Layers 12/16/20/24:
  α ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 5} × β ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 4}; all other layers:
  α ∈ {0.1, 0.2, 0.5, 1, 2} × β ∈ {1, 1.25, 1.5, 2, 4} — 796 single-layer configs/emotion.
  (Blocks L16-22 and L18-24 were also swept but excluded from the single-layer benchmark.)

### 3.6 Interpolation conceptor (Triantafyllopoulos et al., arXiv:2605.04980)
- **Update:** `h' = (1 − β) h + β · C_e h`, `C_e = C(R_e, α)`, applied to **all token positions**
  (whole prompt at prefill, then each generated token), single layer at a time.
- Differences from their protocol: we fit on the emotional pole only (they pool both poles of a
  bipolar concept), and we hook a layer's output (= their pre-block residual of the next layer).
- **Grid:** layers {4, 8, 10, 12, 14, 16, 18, 20, 24} × α ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10} × β ∈ {0.2, 0.4, 0.6, 0.8}.
- Run on the selection set only (not in the held-out benchmark).

### 3.7 Historical: the original repo conceptor (`conceptor_emotion_evaluation.py`)
`h' = h + β · C(h − μ)` at all 28 layers with μ, C from the prompt's last token. Its linear part
`I + βC` has eigenvalues up to 1 + β, so the norm compounds across layers (173 → ~1.1×10⁷ at
layer 27 in our measurement) — 100% of its outputs were degenerate. Kept only as a failure case.

### 3.8 Not yet run: the NeurIPS 2025 appendix protocol (same authors as 3.5)
`x' = x + β·C x` and contrastive `x' = x + β·(C₊ ∧ ¬C₋) x` at the **post-attention layernorm
output** (the normalised MLP input) of one layer, β ∈ {1, 1.5, 2, 10, 50}. Listed for completeness.

---

## 4. Automatic metrics and the selection rule

| Metric | Definition |
|---|---|
| Target confidence | `facebook/bart-large-mnli` zero-shot probability of the target emotion among {joy, anger, disgust, fear, sadness, neutral} |
| Perplexity | Perplexity of the reply under the **unsteered** model, given the same prompt |
| Cross-prompt overlap | Mean pairwise share of word 3-grams between replies to *different* prompts (unsteered ≈ 0.02); catches a config answering every prompt with one template |

**Selection rule (identical for every method):** on the selection set, the configuration with the
highest mean target confidence among those with median perplexity ≤ 50 and cross-prompt
overlap ≤ 0.10. The classifier alone rewards incoherent emotion keywords and templates, hence the
two constraints.

**Selected configurations (benchmark/selection.json):**

| Emotion | Ours (shift+conceptor) | Best single-layer βCh | Plain shift | Project EV |
|---|---|---|---|---|
| joy | L18-24, α=0.15, λ=0.3 | layer 15, α=0.2, β=2 | L16-22, λ=1 | α=3 |
| anger | L14-20, α=0.15, λ=0.25 | layer 12, α=0.5, β=4 | L8-14, λ=1.5 | α=3 |
| disgust | L8-20, α=1, λ=0.3 | layer 13, α=0.2, β=2 | L18-24, λ=2 | α=2 |
| fear | L8-20, α=0.3, λ=0.2 | layer 18, α=0.2, β=1.25 | L10-14, λ=2 | α=3 |
| sadness | L14-20, α=0.1, λ=0.25 | layer 16, α=0.2, β=2 | L18-24, λ=2 | α=3 |

---

## 5. LLM-judge evaluation (human-like scoring)

Scores the held-out benchmark outputs (`benchmark/test/scored.csv`: 5 emotions × 5 systems × 100
prompts = 2,500 rows, **2,058 unique replies** because the baseline reply is shared across emotions).

| Setting | Value |
|---|---|
| Judge model | `gpt-5.5` (OpenAI API), `reasoning_effort=low`, JSON response format |
| Blinding | The judge sees only the user message and the reply — never the method, the configuration, or which emotion was targeted |
| Scoring | Every reply gets coherence, relevance and the intensity of **all five** emotions, so the target emotion is scored without being revealed and off-target leakage is measured |
| Order | Replies shuffled (fixed seed) across methods and emotions |
| Output | `benchmark/judge/judgements.jsonl` (raw), `judged_benchmark.csv` (merged), `judge_table.csv`, `judge_paired.csv` |

### 5.1 Judge system prompt (verbatim)

```
You are a careful human annotator evaluating replies written by an AI assistant.
You will see the user's message and the assistant's reply. Rate the reply on the scales below,
the way an attentive native English reader would. Judge only what is on the page.

1. coherence (1-10): Is the reply fluent, grammatical, well-formed English that reads as a sensible
   response to the user's message? 10 = completely natural and clear. 5 = understandable but with
   clear errors, odd phrasing or partial nonsense. 1 = gibberish, word salad, broken tokens,
   non-English characters, or the same word/phrase repeated over and over.
2. relevance (1-10): Does the reply engage with what the user actually said or asked?
   10 = directly addresses it. 5 = loosely related. 1 = ignores the message entirely.
3. emotion intensity (0-10 for EACH of joy, anger, disgust, fear, sadness): How strongly does the
   reply itself express that emotion, through its tone, word choice and imagery?
   0 = not present at all. 3 = mild hint. 6 = clearly present. 10 = overwhelming, unmistakable.
   Rate each emotion independently; several can be present at once, and a neutral reply scores 0
   on all five. Do not reward a reply for merely naming an emotion word; judge the emotion
   actually conveyed. If the reply is unintelligible (coherence <= 2), rate every emotion 0.

Return ONLY a JSON object with exactly these keys:
{"coherence": <1-10>, "relevance": <1-10>, "joy": <0-10>, "anger": <0-10>, "disgust": <0-10>,
 "fear": <0-10>, "sadness": <0-10>}
```

### 5.2 Judge user message template (verbatim)

```
User message:
"""
{query}
"""

Assistant reply:
"""
{response}
"""
```

### 5.3 Aggregated measures

| Measure | Definition |
|---|---|
| Target intensity | Judge's 0–10 intensity for the emotion that was steered |
| Coherence / relevance | Judge's 1–10 scores |
| Target × coherence | `target × coherence / 10` — emotion delivered in usable text (cf. the Behaviour-Coherence Score of Soo et al. 2025, used in the NeurIPS 2025 conceptor appendix) |
| Off-target intensity | Mean intensity of the four non-target emotions (leakage) |
| Paired comparison | Per prompt, ours minus each rival; mean with 95% bootstrap CI (10,000 resamples), win and tie rates, per emotion and pooled |

**Caveat:** an LLM judge approximates, but does not replace, human ratings. A blinded human rating
sheet for the same outputs can be produced with `make_review.py`.

---

## 6. Relation to the single-word tasks (antonyms, capitalize)

The `conceptor_antonym/` experiments test the same family of operators in a different regime:

| | Single-word tasks (`conceptor_antonym/`) | Open-ended emotion (this folder) |
|---|---|---|
| Task | Zero-shot next-token prediction (`"without:"` → `"With"`) | 40-token free-form reply |
| Fitting data | 100 few-shot prompts; **one row per prompt** (last token) | 250 minimal pairs; **one row per reply token** (~15k rows) |
| Methods | addition `h + βv`, βCh, residual `h + λ(βC − I)h`, mean-corrected `h + λ(βC − I)(h − μ)` | §3 above |
| Layers | **One layer per configuration** (`blocks.ℓ.hook_resid_pre`) | Single layers (βCh, Interpolation) and **stacked blocks** (shift, ours) |
| Steered positions | Last prompt token, one forward pass | Every generation step |
| Score | Exact match of the first output token | Emotion confidence / judge intensity, fluency, relevance |

The stacked shift+conceptor rule has **not** been run on the single-word tasks. There, a single
forward pass decides the answer, the target is one token rather than a style spread across a
reply, and the fitting rows (100) are fewer than the dimension (1536), so the covariance-based
correction is poorly determined. Testing it there would be a useful generality check.
