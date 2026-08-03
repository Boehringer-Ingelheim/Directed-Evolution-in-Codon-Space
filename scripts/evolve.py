# evolve.py
# -*- coding: utf-8 -*-
"""
Greedy synonymous-only codon optimization guided by SynCodonLM (v2).

Encoding matches the validated pipeline:
  - Per-codon tokenizer.convert_tokens_to_ids
  - Manual [BOS, ...body..., EOS] wrapping
  - logits[i + 1] read-out (offset by BOS)
  - token_type_ids only if the tokenizer advertises them

Model, tokenizer, and config are pulled from the Hugging Face Hub on
first use (and cached locally by transformers). No local path is needed.

Exposes:
  - evo_score(): mean masked-LM pseudo-log-likelihood, reported both
    with and without the synonymous constraint.
  - scan_mutations(): single-pass, no-context-update readout of every
    preferred synonymous single-point mutation on the parental background
    (codon-space analogue of Hie et al., Nat Biotechnol 2024).
  - optimize(): greedy directed-evolution walk under a mutation budget.
"""
from __future__ import annotations

import math
import os
from collections import deque

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .codon_utils import (
    DEFAULT_SYNONYMOUS_CODONS,
    ChangeStep,
    EvoScore,
    OptimizationResult,
    SingleMutation,
    build_codon_to_aa,
    clean_dna,
    dna_to_codons,
    hash_ids,
)

# -------------------------------------------------------------------
# SynCodonLM v2 identity on the Hugging Face Hub.
# Override with the SYNCODONLM_MODEL_ID environment variable if you need
# a fork, a gated mirror, or a pinned commit.
# -------------------------------------------------------------------
SYNCODONLM_MODEL_ID = os.environ.get(
    "SYNCODONLM_MODEL_ID",
    "Boehringer-Ingelheim/SynCodonLM-v2",
)


class Evolver:
    """Wrap SynCodonLM v2 and perform greedy synonymous-only edits on DNA
    coding sequences under a user-defined mutation budget.

    On construction, downloads (or reuses the transformers cache for) the
    tokenizer, config, and MLM weights from Hugging Face.
    """

    def __init__(
        self,
        model_id=SYNCODONLM_MODEL_ID,
        device=None,
        dtype=None,
        hf_token=None,
        cache_dir=None,
        revision=None,
        verbose=True,
    ):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Allow token via arg, then env var, then anonymous.
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

        hf_kwargs = {}
        if token is not None:
            hf_kwargs["token"] = token
        if cache_dir is not None:
            hf_kwargs["cache_dir"] = cache_dir
        if revision is not None:
            hf_kwargs["revision"] = revision

        if verbose:
            print(f"[Evolver] Loading SynCodonLM v2 from Hugging Face: {model_id}")
            if revision:
                print(f"[Evolver]   revision: {revision}")
            print(f"[Evolver]   device: {self.device}   dtype: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, **hf_kwargs
        )
        self.config = AutoConfig.from_pretrained(model_id, **hf_kwargs)
        self.model = (
            AutoModelForMaskedLM
            .from_pretrained(
                model_id,
                config=self.config,
                torch_dtype=self.dtype,
                **hf_kwargs,
            )
            .to(self.device)
            .eval()
        )
        self.model_id = model_id

        # Special token ids (fall back to CLS/SEP if BOS/EOS not set).
        self.bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None or self.bos_id is None or self.eos_id is None:
            raise ValueError(
                "SynCodonLM tokenizer must expose BOS/CLS, EOS/SEP, and MASK tokens."
            )

        self.unk_id = self.tokenizer.unk_token_id
        self.supports_tti = "token_type_ids" in getattr(
            self.tokenizer, "model_input_names", []
        )

        # AA -> list of codon token IDs (in the model's vocab, deduped of UNK).
        self._aa_to_ids = {}
        for aa, codons in DEFAULT_SYNONYMOUS_CODONS.items():
            ids = []
            for c in codons:
                tid = self.tokenizer.convert_tokens_to_ids(c)
                if tid is not None and tid != self.unk_id:
                    ids.append(int(tid))
            self._aa_to_ids[aa] = ids

        self._codon_to_aa = build_codon_to_aa(DEFAULT_SYNONYMOUS_CODONS)

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(
                f"[Evolver] Loaded. Parameters: {n_params/1e6:.1f}M   "
                f"token_type_ids supported: {self.supports_tti}"
            )

    # ------------------------------------------------------------------
    # Encoding / decoding (matches the validated pipeline)
    # ------------------------------------------------------------------

    def _in_vocab_synonyms(self, aa):
        """Return (codons, token_ids) for every synonym of `aa` present in vocab."""
        candidate_codons = DEFAULT_SYNONYMOUS_CODONS.get(aa, [])
        codons_out = []
        ids_out = []
        for c in candidate_codons:
            tid = self.tokenizer.convert_tokens_to_ids(c)
            if tid is not None and tid != self.unk_id:
                codons_out.append(c)
                ids_out.append(int(tid))
        return codons_out, ids_out

    def _encode_codons_to_ids(self, codons):
        """Per-codon convert_tokens_to_ids. Raises on any unknown codon."""
        ids = []
        for i, c in enumerate(codons):
            tid = self.tokenizer.convert_tokens_to_ids(c)
            if tid is None or tid == self.unk_id:
                raise ValueError(
                    f"Codon '{c}' at position {i} is not in the tokenizer vocab."
                )
            ids.append(int(tid))
        return ids

    def _ids_to_codons(self, ids):
        """Inverse of _encode_codons_to_ids. Enforces DNA alphabet."""
        toks = self.tokenizer.convert_ids_to_tokens(ids)
        out = []
        for t in toks:
            t = (t or "").replace("U", "T").upper()
            if len(t) != 3:
                raise ValueError(f"Decoded token '{t}' is not a length-3 codon.")
            out.append(t)
        return out

    def _inputs_with_single_mask(self, seq_ids, mask_pos, species_tti):
        """Build the model input dict with position `mask_pos` masked.

        The input is wrapped as [BOS] + body + [EOS] exactly as in the
        validated pipeline, so the codon at body index i lives at
        input_ids[i + 1].
        """
        if not (0 <= mask_pos < len(seq_ids)):
            raise IndexError("mask_pos out of range")
        body = list(seq_ids)
        body[mask_pos] = self.mask_id
        input_ids = torch.tensor(
            [[self.bos_id] + body + [self.eos_id]],
            dtype=torch.long,
            device=self.device,
        )
        attn = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attn}
        if self.supports_tti and species_tti is not None:
            out["token_type_ids"] = torch.full_like(input_ids, int(species_tti))
        return out

    # ------------------------------------------------------------------
    # Shared per-position logits generator
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _perposition_logits(self, seq_ids, codons, token_type_id):
        """Yield per-position logits for every codon position.

        Yields tuples of:
            (i, current_codon, aa, syn_codons_in_vocab,
             syn_token_ids_in_vocab, logits_row_at_i)
        where logits_row_at_i is the raw logits vector at position i,
        already offset for the leading BOS. Used by both evo_score and
        scan_mutations so they can never disagree about how logits are
        produced.
        """
        L = len(seq_ids)
        for i in range(L):
            cur_codon = codons[i]
            aa = self._codon_to_aa[cur_codon]
            syn_codons_v, syn_ids_v = self._in_vocab_synonyms(aa)

            inp = self._inputs_with_single_mask(seq_ids, i, token_type_id)
            logits = self.model(**inp).logits.squeeze(0)   # shape [L+2, |V|]
            row = logits[i + 1]                             # offset by BOS
            yield i, cur_codon, aa, syn_codons_v, syn_ids_v, row

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def evo_score(self, dna, token_type_id=None):
        """Mean masked-LM pseudo-log-likelihood of a CDS, reported both
        with and without the synonymous constraint. Returns an EvoScore."""
        dna = clean_dna(dna)
        codons = dna_to_codons(dna)
        seq_ids = self._encode_codons_to_ids(codons)

        total_syn = 0.0
        total_unc = 0.0
        n = 0
        for i, cur_codon, aa, syn_codons_v, syn_ids_v, row in \
                self._perposition_logits(seq_ids, codons, token_type_id):
            cur_tok = seq_ids[i]

            # Full-vocab softmax -> unconstrained score at this position.
            log_probs_full = F.log_softmax(row, dim=-1)
            total_unc += float(log_probs_full[cur_tok].item())

            # Synonym-restricted softmax -> constrained score at this position.
            if cur_tok not in syn_ids_v:
                # Current codon not among the model's known synonyms; skip.
                continue
            syn_ids_t = torch.tensor(syn_ids_v, dtype=torch.long, device=self.device)
            syn_logits = row.index_select(0, syn_ids_t)
            log_probs_syn = F.log_softmax(syn_logits, dim=-1)
            cur_idx_in_syn = syn_ids_v.index(cur_tok)
            total_syn += float(log_probs_syn[cur_idx_in_syn].item())
            n += 1

        n = max(n, 1)
        return EvoScore(
            synonymous_constrained=total_syn / n,
            unconstrained=total_unc / n,
        )

    # ------------------------------------------------------------------
    # Single-pass mutation scan (parental background, no context updates)
    # ------------------------------------------------------------------

    def scan_mutations(self, dna, token_type_id=None, min_delta_nats=0.0, sort=True):
        """Deep-mutational-scan-style pass over the parental CDS.

        For every codon position:
          1. Mask the position.
          2. Get SynCodonLM's logits.
          3. Restrict to synonyms of the encoded amino acid.
          4. Record the model's preferred synonym and the log-odds margin
             delta = log P(best syn) - log P(current codon).

        No edits are applied and no context updates happen between
        positions - this is a one-shot readout of every high-confidence
        single synonymous swap the model would make on the parental
        background, directly analogous to Hie et al.'s per-residue
        amino-acid scan (Nat Biotechnol 2024) but in codon space.

        Returns a list of SingleMutation.
        """
        dna = clean_dna(dna)
        codons = dna_to_codons(dna)
        seq_ids = self._encode_codons_to_ids(codons)

        results = []
        for i, cur_codon, aa, syn_codons_v, syn_ids_v, row in \
                self._perposition_logits(seq_ids, codons, token_type_id):

            # Nothing to swap for single-codon amino acids or stop.
            if len(syn_ids_v) < 2 or aa in ("M", "W", "*"):
                continue

            cur_tok = seq_ids[i]
            if cur_tok not in syn_ids_v:
                continue
            cur_idx = syn_ids_v.index(cur_tok)

            syn_ids_t = torch.tensor(syn_ids_v, dtype=torch.long, device=self.device)
            syn_logits = row.index_select(0, syn_ids_t)
            log_probs = F.log_softmax(syn_logits, dim=-1)
            order = torch.argsort(log_probs, descending=True).tolist()

            best_rel = order[0]
            if best_rel == cur_idx:
                continue  # current codon already the model's favorite

            best_codon = syn_codons_v[best_rel]
            delta = float((log_probs[best_rel] - log_probs[cur_idx]).item())
            if delta < min_delta_nats:
                continue

            if len(order) > 1:
                runner_rel = order[1]
                runner_codon = syn_codons_v[runner_rel]
                margin_2nd = float(
                    (log_probs[best_rel] - log_probs[runner_rel]).item()
                )
            else:
                runner_codon = "---"
                margin_2nd = float("inf")

            results.append(SingleMutation(
                pos_1based=i + 1,
                aa=aa,
                from_codon=cur_codon,
                to_codon=best_codon,
                runner_up_codon=runner_codon,
                delta_nats=delta,
                odds_ratio=math.exp(delta),
                margin_to_2nd_nats=margin_2nd,
            ))

        if sort:
            results.sort(key=lambda m: -m.delta_nats)
        return results

    # ------------------------------------------------------------------
    # Greedy directed-evolution walk
    # ------------------------------------------------------------------

    def optimize(
        self,
        sequences,
        token_type_id=None,
        max_change_fraction=0.10,
        min_delta_nats=1e-9,
        min_delta_nats_at_cap=0.10,
        lock_start=True,
        lock_stop=True,
        tabu_size=32,
        max_iterations=None,
    ):
        """Iterative greedy synonymous refinement guided by SynCodonLM.

        The change budget is defined against the ORIGINAL parental CDS
        (not the running one), matching the paper's convention.

        Returns a list of OptimizationResult.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        results = []

        for idx, dna in enumerate(sequences):
            dna_orig = clean_dna(dna)
            codons_orig = dna_to_codons(dna_orig)
            seq_ids_orig = self._encode_codons_to_ids(codons_orig)
            L = len(seq_ids_orig)

            change_cap = int(math.floor(max_change_fraction * L))
            iter_cap = (
                max_iterations if max_iterations is not None
                else max(2 * change_cap, 1)
            )

            orig_score = self.evo_score(dna_orig, token_type_id)

            seq_ids_cur = list(seq_ids_orig)
            trace = []
            visited = deque(maxlen=tabu_size)
            visited.append(hash_ids(seq_ids_cur))

            stop_reason = "no improving move"
            it = 0

            while it < iter_cap:
                codons_cur = self._ids_to_codons(seq_ids_cur)
                changed_now = sum(
                    1 for a, b in zip(seq_ids_cur, seq_ids_orig) if a != b
                )
                at_cap = changed_now >= change_cap
                delta_floor = (
                    min_delta_nats_at_cap if at_cap else min_delta_nats
                )

                best = None
                # (delta, i, from_tok, to_tok, from_c, to_c, runner_c, margin_2nd)

                for i, cur_codon, aa, syn_codons_v, syn_ids_v, row in \
                        self._perposition_logits(seq_ids_cur, codons_cur, token_type_id):

                    if lock_start and i == 0:
                        continue
                    if lock_stop and i == L - 1:
                        continue
                    if len(syn_ids_v) < 2 or aa in ("M", "W", "*"):
                        continue

                    cur_tok = seq_ids_cur[i]
                    if cur_tok not in syn_ids_v:
                        continue
                    cur_idx = syn_ids_v.index(cur_tok)

                    syn_ids_t = torch.tensor(
                        syn_ids_v, dtype=torch.long, device=self.device
                    )
                    syn_logits = row.index_select(0, syn_ids_t)
                    log_probs = F.log_softmax(syn_logits, dim=-1)
                    order = torch.argsort(log_probs, descending=True).tolist()

                    best_rel = order[0]
                    if best_rel == cur_idx:
                        continue

                    # Budget guard: at cap, don't create new deviations
                    # from parent; allow only reverts or refinements at
                    # already-changed positions.
                    parent_tok = seq_ids_orig[i]
                    top_tok = syn_ids_v[best_rel]
                    if at_cap and cur_tok == parent_tok and top_tok != parent_tok:
                        continue

                    delta = float(
                        (log_probs[best_rel] - log_probs[cur_idx]).item()
                    )
                    if delta < delta_floor:
                        continue

                    top_codon = syn_codons_v[best_rel]

                    if len(order) > 1:
                        runner_rel = order[1]
                        runner_codon = syn_codons_v[runner_rel]
                        margin_2nd = float(
                            (log_probs[best_rel] - log_probs[runner_rel]).item()
                        )
                    else:
                        runner_codon = "---"
                        margin_2nd = float("inf")

                    if (best is None) or (delta > best[0]):
                        best = (
                            delta, i, cur_tok, top_tok,
                            cur_codon, top_codon, runner_codon, margin_2nd,
                        )

                if best is None:
                    stop_reason = (
                        f"stopped: change-cap reached ({changed_now}/{L})"
                        if at_cap else "no improving move"
                    )
                    break

                delta, i, from_tok, to_tok, from_c, to_c, runner_c, margin_2nd = best

                trial = list(seq_ids_cur)
                trial[i] = to_tok
                trial_hash = hash_ids(trial)
                if trial_hash in visited:
                    stop_reason = "stopped: cycle/oscillation detected"
                    break
                visited.append(trial_hash)

                seq_ids_cur = trial
                new_changed = sum(
                    1 for a, b in zip(seq_ids_cur, seq_ids_orig) if a != b
                )
                trace.append(ChangeStep(
                    iter=it,
                    pos_1based=i + 1,
                    from_codon=from_c,
                    to_codon=to_c,
                    runner_up_codon=runner_c,
                    delta_nats=delta,
                    odds_ratio=math.exp(delta),
                    margin_to_2nd_nats=margin_2nd,
                    changed_positions_vs_orig=new_changed,
                    change_cap_allowed=change_cap,
                ))

                it += 1

            dna_final = "".join(self._ids_to_codons(seq_ids_cur))
            final_score = self.evo_score(dna_final, token_type_id)
            positions_changed = sum(
                1 for a, b in zip(seq_ids_cur, seq_ids_orig) if a != b
            )

            results.append(OptimizationResult(
                input_index=idx,
                original_dna=dna_orig,
                optimized_dna=dna_final,
                length_codons=L,
                positions_changed=positions_changed,
                original_evo_score=orig_score.synonymous_constrained,
                optimized_evo_score=final_score.synonymous_constrained,
                delta_evo_score=(
                    final_score.synonymous_constrained
                    - orig_score.synonymous_constrained
                ),
                stop_reason=stop_reason,
                trace=trace,
                original_evo_score_unconstrained=orig_score.unconstrained,
                optimized_evo_score_unconstrained=final_score.unconstrained,
            ))

        return results