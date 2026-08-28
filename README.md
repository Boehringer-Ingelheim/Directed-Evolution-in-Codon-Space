# Directed Evolution in Codon Space

**Making a few targeted synonymous mutations to already-optimized coding sequences can make them significantly better.** We use synonymous edits guided by **SynCodonLM**, applied one at a time in updated context, pushing a coding sequence up a fitness gradient defined by the language model. The protein sequence never changes.

The premise is simple: protein engineers iteratively refine functional proteins into better versions of themselves, but codon optimization has stayed stuck in an older paradigm where a coding sequence is generated once and frozen. This repo instead treats a pre-optimized parental CDS as a starting point to be evolved, letting SynCodonLM propose a small number of synonymous edits, each conditioned on the sequence as it currently stands.

---

## Install

```bash
git clone https://github.com/Boehringer-Ingelheim/Directed-Evolution-in-Codon-Space.git
cd Directed-Evolution-in-Codon-Space
pip install -e .
pip install -r requirements.txt
```

Python 3.9+. GPU optional but recommended for long CDS. Model weights are pulled from Hugging Face automatically on first run (cached under `~/.cache/huggingface/`). If the SynCodonLM repo is gated for your account, log in first with `huggingface-cli login` or `export HF_TOKEN=hf_xxx`.

---

## Three modes

The `Evolver` class exposes three entry points. Run from inside the `scripts/` directory (or add it to your `PYTHONPATH`) so the flat imports resolve.

| Mode | Question it answers | Method |
|---|---|---|
| **Score** | How host-like is this CDS? | `evolver.evo_score(dna, token_type_id)` |
| **Scan** | Which single synonymous edits does the model prefer on the parental background? | `evolver.scan_mutations(dna, token_type_id)` |
| **Evolve** | Refine my parental CDS under a mutation budget, updating context after each edit. | `evolver.optimize([dna], token_type_id, max_change_fraction=0.10)` |

Use **`scan_mutations`** for a static shortlist of individually-preferred edits (e.g. site-directed synonymous mutagenesis). Use **`optimize`** to actually produce an evolved CDS — each accepted edit reshapes the model's preferences at neighboring positions. This is the paper's main procedure. A runnable script demonstrating all three lives at `scripts/example.py`.

---

## Quickstart

```python
from evolve import Evolver
from codon_utils import print_optimization_report

evolver = Evolver()   # loads SynCodonLM v2 from Hugging Face

results = evolver.optimize(
    ["ATG...TAA"],               # your parental CDS(s) — single string or list
    token_type_id=345,           # CHO — see the table below
    max_change_fraction=0.10,    # 10% mutation cap, matching the paper
)
print_optimization_report(results)
```

The report lists each accepted synonymous swap (position, from→to, Δ in nats, odds ratio) and the change in **Evo score** — the mean masked-LM pseudo-log-likelihood, where higher means the host's codon "language model" finds the sequence more plausible.

To just score a sequence:

```python
score = evolver.evo_score("ATG...TAA", token_type_id=345)
print(score.synonymous_constrained)   # what the walk optimizes (protein held fixed)
print(score.unconstrained)            # full-vocab; sensitive to AA identity
```

To get a ranked readout of every preferred single-point synonymous mutation on the parental background:

```python
from codon_utils import print_mutation_scan

hits = evolver.scan_mutations("ATG...TAA", token_type_id=345, min_delta_nats=0.10)
print_mutation_scan(hits, top=20)
```

---

## Key arguments

| Argument | What it does | Typical |
|---|---|---|
| `max_change_fraction` | Cap on % of codons that may differ from the parental sequence | `0.10` (paper default) |
| `token_type_id` | Host/species id (see table below) | `345` (CHO) |
| `min_delta_nats` | Minimum per-edit log-odds improvement to report/apply | `0.10` for shortlists, `0.0` for a full scan |

---

## Common host token-type IDs

Full species map: [`species_token_type.py`](https://github.com/Boehringer-Ingelheim/SynCodonLM/blob/master/SynCodonLM/species_token_type.py)

| Organism | Token-Type ID |
|---|:-:|
| *C. griseus* (CHO — the paper's host) | **345** |
| *H. sapiens* | **373** |
| *M. musculus* | 368 |
| *D. rerio* | 428 |
| *D. melanogaster* | 190 |
| *C. elegans* | 212 |
| *S. cerevisiae* | 118 |
| *E. coli* | 30 |
| *A. thaliana* | 258 |

---

## Citation

If you use this repo, please cite:

```bibtex
@article{Heuschkel2026DirectedEvolution,
    author = {Heuschkel, James and Kingsley, Laura and Reed, Jon and Li, Di and Warner, Matthew and Pefaur, Noah and Cramer, Steven},
    title = {Directed Evolution in Codon Space},
    journal = {bioRxiv},
    year = {2026},
    elocation-id = {2026.08.03.742557},
    doi = {10.64898/2026.08.03.742557},
    publisher = {Cold Spring Harbor Laboratory},
    url = {https://doi.org/10.64898/2026.08.03.742557}
}
```

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
