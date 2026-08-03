![Codon Evolution Logo](logo/logo.png)

# Directed Evolution in Codon Space

**This work shows how making a few targeted synonymous mutations to already-optimized coding sequences can make them significantly better.**
We use synonymous edits guided by **SynCodonLM**, applied one at a time in updated context, pushing the sequence up a fitness gradient defined by the language model.

---

## The premise

In protein engineering, functional proteins are iteratively refined into better and better versions of themselves. We asked why this idea hasn't been applied to coding DNA. Codon optimization has stayed stuck in an older paradigm: a coding sequence is generated once and frozen.

**This repo treats coding sequences the way protein engineers treat proteins.**
Start from a coding sequence that already works — a pre-optimized parental CDS — and let SynCodonLM propose a small number of synonymous edits, one at a time, each conditioned on the sequence as it currently stands. The protein sequence never changes.

---

## What "directed evolution in codon space" means, concretely

At each step, the algorithm:

1. **Masks one codon position** in the current sequence.
2. Asks SynCodonLM for its logits over the vocabulary.
3. **Restricts to the synonyms** of the amino acid encoded at that position — the protein never changes.
4. Computes a **head-to-head Δ margin** between the model's preferred synonym and the codon currently there.
5. Across every eligible position, keeps the **single largest Δ** and applies it.
6. Rescores the updated sequence and repeats.

Stopping conditions:

- **Mutation budget hit.** By default, no more than 10% of codon positions may differ from the parental sequence — matching the paper's cap and preventing large excursions from a validated starting point.
- **No admissible Δ remains** (nothing improves the local log-odds under threshold).
- **Cycle or immediate reversal detected** — a tabu memory and state-hash check prevent the greedy walk from oscillating.
- **Boundary preservation.** Start ATG and terminal stop are locked by default.

The "Evo score" reported alongside every result is the mean masked-LM pseudo-log-likelihood (in nats): higher = the host's codon "language model" finds the sequence more plausible.

---

## Install

```bash
git clone https://github.com/Boehringer-Ingelheim/Directed-Evolution-in-Codon-Space.git
cd Directed-Evolution-in-Codon-Space
pip install -r requirements.txt
```

Python 3.9+. GPU optional but recommended for long CDS.

**Weights are pulled from Hugging Face automatically** the first time you run anything (and cached under `~/.cache/huggingface/`). If the SynCodonLM repo is gated for your account, log in first:

```bash
huggingface-cli login
# or:
export HF_TOKEN=hf_xxx
```

To point at a fork, a pinned commit, or a mirror without touching code:

```bash
export SYNCODONLM_MODEL_ID="your-fork/SynCodonLM-v2"
```

---

## Three modes at a glance

The `Evolver` class exposes three entry points, matching three natural questions:

| Mode | Question it answers | Method |
|---|---|---|
| **Score** | *How host-like is this CDS?* | `evolver.evo_score(dna, token_type_id)` |
| **Scan** | *If I could make just N single synonymous edits, which ones?* | `evolver.scan_mutations(dna, token_type_id)` |
| **Evolve** | *Refine my parental CDS under a mutation budget, updating context after each edit.* | `evolver.optimize([dna], token_type_id, max_change_fraction=0.10)` |

A single runnable script demonstrating all three lives at `scripts/example.py`.

---

## Quickstart

Run from inside the `scripts/` directory (or add it to your `PYTHONPATH`) so the flat imports resolve correctly:

```python
from evolve import Evolver
from codon_utils import print_optimization_report

evolver = Evolver()   # loads SynCodonLM v2 from Hugging Face

results = evolver.optimize(
    ["ATG...TAA"],              # your parental CDS(s) — single string or list
    token_type_id=345,           # CHO — see the table below
    max_change_fraction=0.10,    # 10% mutation cap, matching the paper
)
print_optimization_report(results)
```

---

## What you get back from `optimize()`

```
================================================
 Sequence #0  (L = 447 codons)
================================================
Original  evo score (syn)      :  -1.842103 nats   <- parental / starting point
Optimized evo score (syn)      :  -1.615887 nats   <- after directed evolution
Delta evo score (syn)          : +0.226216 nats (odds x 1.254)
Original  evo score (unconstr.):  -3.204518 nats
Optimized evo score (unconstr.):  -3.198744 nats
Positions changed              :  42 / 447 (9.4%)
Stop reason                    : stopped: change-cap reached (44/447)

Preferred changes (42):
  iter    pos   from -> to      delta (nats)     odds   runner-up
     0    118    CTG -> CTC         0.9821    2.670         CTT
     1     73    GCG -> GCC         0.7112    2.037         GCT
     ...
```

Each row is one accepted synonymous swap: the model preferred it over what was there, and the walk applied it in the updated context. `runner-up` is the second-best synonym at that position — a quick read on how confident the pick was.

The **unconstrained** evo score is included as a sanity check: a synonymous-only rewrite shouldn't drift much in unconstrained score (the protein hasn't changed), so a large drop there usually indicates a codon that fell into a genuinely low-probability region.

---

## Just want to score a sequence?

```python
from evolve import Evolver

evolver = Evolver()
score = evolver.evo_score("ATG...TAA", token_type_id=345)
print(score.synonymous_constrained)   # what the walk optimizes
print(score.unconstrained)            # full-vocab log-softmax; sensitive to AA identity
```

`evo_score` returns an `EvoScore` object with two fields, both in nats:

| Field | Softmax normalized over | What it tells you |
|---|---|---|
| `synonymous_constrained` | Only the synonyms of the AA at each position | How host-like the *codon choice* is, holding the protein fixed. This is the objective the directed-evolution walk climbs. |
| `unconstrained` | The full codon vocabulary | How plausible the sequence is *globally*, including amino-acid identity. Useful as a sanity check that a synonymous rewrite hasn't wandered into a globally-unlikely regime, and for cross-protein comparisons of similar length. |

`float(score)` returns the synonymous-constrained value for back-compat with older scripts.

Both flavors are useful for:

- **Ranking candidates** before spending on synthesis.
- **Diagnosing why** a parental construct expresses poorly.
- **Tracking evolution over time** — in the paper, Evo scores on H1N1 nucleoprotein, hemagglutinin, and neuraminidase coding sequences increased with sampling year (r = 0.65, 0.79, 0.41; all *p* < 1e-3), while ESM-2 scores on the *same* sequences translated to protein went the opposite direction. Codon-level likelihood carries evolutionary signal that amino-acid-level likelihood doesn't.

---

## One-shot scan: every preferred synonymous single-point mutation

If you don't want the greedy iterative walk and just want a **ranked readout of every synonymous single-point mutation SynCodonLM would prefer on the parental background** — the codon-space analogue of the per-residue scan in *Efficient evolution of human antibodies from general protein language models* (Hie et al., Nat Biotechnol 2024) — use `scan_mutations`:

```python
from evolve import Evolver
from codon_utils import print_mutation_scan

evolver = Evolver()

hits = evolver.scan_mutations(
    "ATG...TAA",
    token_type_id=345,       # CHO
    min_delta_nats=0.10,     # only report clearly-preferred swaps
)

print_mutation_scan(hits, top=20)
```

Example output:

```
   pos  aa  from     to       delta (nats)    odds   runner-up
   118   L   CTG -> CTC         0.9821       2.670    CTT
    73   A   GCG -> GCC         0.7112       2.037    GCT
   241   R   CGG -> CGC         0.6540       1.923    CGT
   ...
```

Each row is one position where the model prefers a different synonym than the parental codon. Every Δ is computed on the **parental** background — the scan does not update context between positions and does not apply any edit. This makes it useful for:

- **Prioritizing site-directed synonymous mutagenesis** — pick the top-N and order those primers.
- **Sanity-checking the walk** — the highest-Δ hit here is exactly the position + swap that `optimize()` picks on iteration 0.
- **Diagnostics** — which codons in a parental CDS the host model most dislikes, and by how much.

Notes:

- Positions encoding **Met, Trp, or stop** are skipped (no synonymous alternative).
- Positions where the current codon is already the model's top pick are omitted.
- The output list is sorted by Δ descending by default; pass `sort=False` to keep positional order.
- To get *every* per-position preference regardless of magnitude, set `min_delta_nats=0.0` (the default).

### When to use which

- **`scan_mutations`** — you want a **static shortlist** of individually-preferred synonymous edits (e.g. for site-directed mutagenesis or for a mutational-scan-style figure). No context updates, no interactions accounted for.
- **`optimize`** — you want a **coherent rewrite trajectory** where each accepted edit reshapes the model's preferences at neighboring positions. This is the paper's main procedure and what you should use to actually produce an "evolved" CDS for expression.

The two agree on iteration 0 by construction. They diverge from iteration 1 onward, because a synonymous change at position 118 shifts the model's preferences at positions 116, 119, 120, etc.

---

## Knobs worth knowing

| Argument | What it does | Typical |
|---|---|---|
| `max_change_fraction` | Cap on % of codons that may differ from the parental sequence | `0.10` (paper default) |
| `token_type_id` | Host/species id (see table below) | `345` (CHO) |
| `min_delta_nats` (in `optimize`) | Minimum per-edit log-odds improvement | `1e-9` |
| `min_delta_nats_at_cap` | Stricter Δ once the cap is hit (avoids micro-swaps) | `0.10` |
| `min_delta_nats` (in `scan_mutations`) | Minimum log-odds Δ to report a preferred swap | `0.10` for shortlists, `0.0` for a full scan |

The full species map lives here:
https://github.com/Boehringer-Ingelheim/SynCodonLM/blob/master/SynCodonLM/species_token_type.py.
A quick reference for common hosts:

| Organism             | Token-Type ID |
|----------------------|:-------------:|
| ***C. griseus*** (CHO — the paper's host) | **345** |
| ***H. sapiens***     | **373** |
| *M. musculus*        | 368 |
| *D. rerio*           | 428 |
| *D. melanogaster*    | 190 |
| *C. elegans*         | 212 |
| *S. cerevisiae*      | 118 |
| *E. coli*            | 30  |
| *A. thaliana*        | 258 |

---

## Design philosophy

Three things make this a *directed evolution* tool rather than a codon optimizer:

1. **Your parental sequence is the anchor.** The budget is defined against the *original* CDS, not the running one. Even if the walk swaps a codon and later reverts it, the budget only counts positions that still differ from the parent.
2. **Every edit is scored in updated context.** A synonymous change at position 118 shifts the model's preferences at positions 116, 119, 120, etc. Batch codon replacement can't see those interactions; iterative single-edit refinement can.
3. **The walk is greedy and interpretable.** Every accepted move is a specific position, a specific from→to swap, a specific Δ, and a specific odds ratio, all recorded in the returned trace. Nothing is hidden.

The 10% cap matters. It's not just a safety rail — it's the mechanism that keeps the refined sequence *biologically adjacent* to a construct that already works. In the paper, this cap preserved product-quality attributes in the vast majority of cases while still delivering a significant titer improvement across the panel.

---

## Repo layout

```
.
├── scripts/
│   ├── example.py        # runnable end-to-end demo of all three modes
│   ├── evolve.py         # Evolver class: HF loading, evo_score, scan_mutations, optimize
│   └── codon_utils.py    # genetic code, DNA helpers, result dataclasses, printers
└── README.md
```

Run examples from inside `scripts/` (or add it to `PYTHONPATH`) so the flat imports (`from evolve import Evolver`, `from codon_utils import ...`) resolve correctly.

---

## Citation

If you use this repo, please cite:

> Heuschkel J., Kingsley L., Reed J., Li D., Warner M., Pefaur N., Cramer S.
> **Directed Evolution in Codon Space.** (2026).

and the underlying model:

```bibtex
@article{10.1093/nar/gkag166,
    author  = {Heuschkel, James and Kingsley, Laura and Pefaur, Noah and Nixon, Andrew and Cramer, Steven},
    title   = {Advancing codon language modeling with synonymous codon constrained masking},
    journal = {Nucleic Acids Research},
    volume  = {54},
    number  = {5},
    pages   = {gkag166},
    year    = {2026},
    month   = {02},
    issn    = {1362-4962},
    doi     = {10.1093/nar/gkag166},
    url     = {https://doi.org/10.1093/nar/gkag166}
}
```

---

## License

MIT (see `LICENSE`).
