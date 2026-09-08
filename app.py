"""
DeepMutate-3D, interactive protein mutation scanner.

Pipeline
--------
1. An ESM-2 protein language model scores every possible point mutation in a
   sequence via a Log-Likelihood Ratio (LLR):  log P(mutant) - log P(wildtype).
2. The 19 non-wildtype LLRs at each position are averaged into a single
   "Residue Sensitivity Score".
3. The matching predicted 3D fold is pulled from the AlphaFold DB.
4. The scores are written into the PDB B-factor column and painted onto the
   ribbon as a red (fragile) -> blue (tolerant) heatmap with py3Dmol, with
   per-residue hover readouts driven by the full LLR matrix.

Sequences longer than the ESM-2 context are scored with an overlapping sliding
window, so there is no hard length limit.

Runs on CUDA (Hugging Face ZeroGPU), Apple Silicon `mps`, or CPU. Weights are
placed at import time as ZeroGPU expects; per-call tensors are built inside the
`@spaces.GPU` context.
"""

from __future__ import annotations

import base64
import html as html_lib
import inspect
import json
import os
import pathlib
import re
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd
import py3Dmol
import requests
import torch

# ---------------------------------------------------------------------------
# ZeroGPU compatibility
# ---------------------------------------------------------------------------
# `spaces.GPU` requests a slice of NVIDIA hardware on Hugging Face ZeroGPU tiers.
# Off-Space (e.g. a local M3 Mac) the decorator is a transparent no-op, and if the
# library is absent altogether we substitute an equivalent pass-through.
try:
    import spaces
except ImportError:  # pragma: no cover - only hit in minimal local envs

    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def _decorator(fn):
                return fn

            return _decorator

    spaces = _SpacesShim()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ESM-2 comes in a size ladder. Larger models score better but cost
# proportionally more compute; the relative cost factors below feed the ZeroGPU
# duration estimate.
MODEL_CHOICES = {
    "ESM-2 35M (fastest)": ("facebook/esm2_t12_35M_UR50D", 0.3),
    "ESM-2 150M (balanced)": ("facebook/esm2_t30_150M_UR50D", 1.0),
    "ESM-2 650M (most accurate)": ("facebook/esm2_t33_650M_UR50D", 4.0),
}
# Benchmarked on ProteinGym: 650M in wildtype-marginal mode (rho 0.431) beats
# 150M in masked-marginal mode (rho 0.399) at a fraction of the compute, so the
# larger model is the default rather than the more expensive scoring protocol.
DEFAULT_MODEL_LABEL = "ESM-2 650M (most accurate)"
MODEL_ID = MODEL_CHOICES[DEFAULT_MODEL_LABEL][0]

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # the 20 standard residues

# ESM-2's context is 1024 tokens; minus <cls>/<eos> that is 1022 residues. Longer
# proteins are covered by overlapping windows rather than being truncated.
WINDOW_RESIDUES = 1022
WINDOW_OVERLAP = 256
# A guard rail, not a model limit: masked-marginal scoring costs one forward pass
# per residue, so very long inputs are refused rather than left spinning.
ABSOLUTE_MAX_RESIDUES = 4000
MASKED_BATCH = 8  # masked-marginal forward passes bundled per batch

# The AlphaFold DB retires old model versions, so we ask the prediction API for
# the current `pdbUrl` and only fall back to hand-built file URLs if it is down.
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"
ALPHAFOLD_FALLBACK_VERSIONS = (6, 5, 4)
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
REQUEST_TIMEOUT = 30

# Standard OS font stacks - no webfont downloads, identical metrics to the rest
# of the desktop. Used by the Gradio theme and every piece of inline HTML.
FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)
MONO_STACK = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace'



LOGO_PATH = pathlib.Path(__file__).parent / "docs" / "assets" / "logo-header.png"


def _logo_data_uri() -> str:
    """Inline the logo so the page needs no static file route.

    Returns an empty string if the asset is missing, so a partial checkout still
    launches rather than failing at import.
    """
    try:
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


LOGO_DATA_URI = _logo_data_uri()

_MODEL_CACHE: dict = {}


def resolve_model(label: Optional[str]) -> tuple:
    """Map a dropdown label to (model repo id, relative cost factor)."""
    return MODEL_CHOICES.get(label or "", MODEL_CHOICES[DEFAULT_MODEL_LABEL])


def resolve_device() -> torch.device:
    """CUDA on ZeroGPU, Metal on Apple Silicon, CPU everywhere else."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device, model_id: str = MODEL_ID):
    """Fetch tokenizer + weights and place them on `device`.

    Cached per (model, device) pair, so switching models in the UI keeps both
    resident rather than re-downloading on every toggle. Idempotent.
    """
    key = (model_id, str(device))
    if key not in _MODEL_CACHE:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model.eval().to(device)
        _MODEL_CACHE[key] = (tokenizer, model)
    return _MODEL_CACHE[key]


def preload_model() -> None:
    """Place the model at import time.

    Hugging Face's ZeroGPU guidance is explicit that model weights should be
    moved to `cuda` at module level, not inside `@spaces.GPU`: a PyTorch CUDA
    emulation layer makes the placement valid before any physical GPU is
    attached, and transfers staged at startup are materialised far more
    efficiently than ones issued mid-call. Lazy loading inside the decorated
    function still works, but is explicitly discouraged.

    Input tensors are a different matter - those are still built inside the GPU
    context, where the real device exists.

    Failure here is not fatal: with no network at boot we fall back to loading
    on first use, which is slower but keeps the UI reachable.
    """
    try:
        load_model(resolve_device(), MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - startup must not be fatal
        print(f"[DeepMutate-3D] Model preload deferred to first scan: {exc}")


# ---------------------------------------------------------------------------
# Sequence handling
# ---------------------------------------------------------------------------
def clean_sequence(raw: str) -> str:
    """Strip FASTA headers, whitespace and gaps; upper-case the residues."""
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if not ln.startswith(">")]
    return re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()


def validate_sequence(seq: str) -> Optional[str]:
    if not seq:
        return "No protein sequence supplied."
    unknown = sorted(set(seq) - set(AMINO_ACIDS) - set("BXZJUO"))
    if unknown:
        return f"Sequence contains non-amino-acid characters: {', '.join(unknown)}"
    return None


def sequence_windows(n_res: int) -> List[Tuple[int, int]]:
    """Split a sequence into overlapping windows that fit the ESM-2 context.

    Each interior residue is scored in two windows and its log-probabilities are
    averaged, which avoids the artefacts you get when a residue happens to sit at
    a hard window edge with no downstream context.
    """
    if n_res <= WINDOW_RESIDUES:
        return [(0, n_res)]
    step = WINDOW_RESIDUES - WINDOW_OVERLAP
    starts = list(range(0, n_res - WINDOW_RESIDUES + 1, step))
    if starts[-1] + WINDOW_RESIDUES < n_res:
        starts.append(n_res - WINDOW_RESIDUES)
    return [(s, s + WINDOW_RESIDUES) for s in starts]


# ---------------------------------------------------------------------------
# External biological APIs
# ---------------------------------------------------------------------------
def fetch_uniprot_sequence(uniprot_id: str) -> Tuple[Optional[str], str]:
    """Pull the canonical sequence for a UniProt accession. Returns (seq, note)."""
    uniprot_id = (uniprot_id or "").strip().upper()
    if not uniprot_id:
        return None, "No UniProt ID supplied."
    url = UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 404:
            return None, f"UniProt has no entry for '{uniprot_id}'."
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, f"UniProt request failed: {exc}"
    seq = clean_sequence(response.text)
    if not seq:
        return None, f"UniProt returned an empty sequence for '{uniprot_id}'."
    return seq, f"Sequence fetched from UniProt ({len(seq)} residues)."


def _alphafold_pdb_url(uniprot_id: str) -> Optional[str]:
    """Ask the AlphaFold API which PDB file is current for this accession."""
    try:
        response = requests.get(
            ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None
    if isinstance(payload, list) and payload:
        entry = payload[0]
        url = entry.get("pdbUrl")
        if url:
            return url
        version = entry.get("latestVersion")
        if version:
            return ALPHAFOLD_PDB_URL.format(uniprot_id=uniprot_id, version=version)
    return None


def fetch_alphafold_pdb(uniprot_id: str) -> Tuple[Optional[str], str]:
    """Fetch AlphaFold DB predicted coordinates. Returns (pdb_text, note).

    Any failure (bad accession, no model, network trouble) is reported as a note
    instead of raising, so the sequence scan still renders.
    """
    uniprot_id = (uniprot_id or "").strip().upper()
    if not uniprot_id:
        return None, "No UniProt ID supplied - skipping structural retrieval."

    candidates = []
    api_url = _alphafold_pdb_url(uniprot_id)
    if api_url:
        candidates.append(api_url)
    candidates.extend(
        ALPHAFOLD_PDB_URL.format(uniprot_id=uniprot_id, version=v)
        for v in ALPHAFOLD_FALLBACK_VERSIONS
    )

    last_note = f"AlphaFold DB has no model for '{uniprot_id}'."
    for url in candidates:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                continue
            response.raise_for_status()
        except requests.RequestException as exc:
            last_note = f"AlphaFold request failed: {exc}"
            continue
        if not response.text.lstrip().startswith(("HEADER", "ATOM", "MODEL")):
            last_note = f"AlphaFold returned an unexpected payload for '{uniprot_id}'."
            continue
        return response.text, f"AlphaFold structure retrieved for {uniprot_id}."
    return None, last_note


# ---------------------------------------------------------------------------
# Core inference, runs on the ZeroGPU allocation
# ---------------------------------------------------------------------------
def _window_log_probs(model, tokenizer, chunk: str, mode: str, device) -> torch.Tensor:
    """Per-residue log-probability distributions for one window. (len, vocab)."""
    encoded = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    n_res = len(chunk)

    # ESM wraps the sequence as <cls> M V L ... <eos>, so residue 0 is token 1.
    # Computed rather than hardcoded: an off-by-one here shifts the whole heatmap
    # by a residue and still looks plausible, which is what makes it dangerous.
    offset = 1 if int(input_ids[0, 0]) == tokenizer.cls_token_id else 0

    if mode.startswith("Wildtype"):
        # One forward pass over the intact sequence: fast, slightly blunter,
        # because the model can see the wildtype residue in its own input.
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        return log_probs[offset : offset + n_res]

    # Masked marginals (Meier et al. 2021): mask each position in turn so the
    # model cannot read the wildtype off the input. n_res passes, batched.
    collected = []
    for start in range(0, n_res, MASKED_BATCH):
        positions = list(range(start, min(start + MASKED_BATCH, n_res)))
        batch = input_ids.repeat(len(positions), 1)
        for row, pos in enumerate(positions):
            batch[row, pos + offset] = tokenizer.mask_token_id
        batch_mask = attention_mask.repeat(len(positions), 1)
        logits = model(input_ids=batch, attention_mask=batch_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, pos in enumerate(positions):
            collected.append(log_probs[row, pos + offset])
    return torch.stack(collected)


def estimate_duration(
    sequence: str,
    mode: str = "Masked marginals (accurate)",
    model_label: Optional[str] = None,
) -> int:
    """Seconds of GPU time to reserve for this specific scan.

    ZeroGPU accepts a callable here, taking the same arguments as the decorated
    function. Reserving only what the job needs matters twice over: shorter
    requests get higher queue priority, and the free daily quota is small enough
    (5 GPU-minutes) that over-reserving is expensive.

    Masked marginals costs one forward pass per residue; wildtype marginals
    costs one per window.
    """
    n_res = len(sequence or "")
    _, cost = resolve_model(model_label)
    if mode.startswith("Wildtype"):
        seconds = 20 + 5 * cost * len(sequence_windows(max(n_res, 1)))
    else:
        seconds = 20 + 0.08 * cost * n_res
    return int(max(30, min(300, seconds)))


@spaces.GPU(duration=estimate_duration)
def compute_sensitivity(
    sequence: str,
    mode: str = "Masked marginals (accurate)",
    model_label: Optional[str] = None,
) -> Tuple[List[float], List[List[float]]]:
    """Log-Likelihood Ratios for every substitution at every position.

    LLR(mutant) = log P(mutant | context) - log P(wildtype | context)

    Returns (sensitivity_scores, llr_matrix) where the matrix is n_res x 20 in
    `AMINO_ACIDS` order and the scores are the mean of the 19 non-wildtype
    columns.

    Every per-call tensor is created inside this function, on the device that
    exists at call time. The model itself is placed at import (see
    `preload_model`); `load_model` here is a cache hit in the normal case and a
    fallback if the preload was skipped.
    """
    device = resolve_device()
    model_id, _ = resolve_model(model_label)
    tokenizer, model = load_model(device, model_id)

    n_res = len(sequence)
    mut_index = torch.tensor(
        tokenizer.convert_tokens_to_ids(AMINO_ACIDS), device=device
    )
    wt_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(sequence)), device=device
    )

    # Accumulators in sequence space; windows add into their own slice.
    mutant_sum = torch.zeros(n_res, len(AMINO_ACIDS), device=device)
    wildtype_sum = torch.zeros(n_res, device=device)
    counts = torch.zeros(n_res, device=device)

    with torch.no_grad():
        for start, end in sequence_windows(n_res):
            residue_lp = _window_log_probs(
                model, tokenizer, sequence[start:end], mode, device
            )
            mutant_sum[start:end] += residue_lp[:, mut_index]
            wildtype_sum[start:end] += (
                residue_lp.gather(1, wt_ids[start:end].unsqueeze(1)).squeeze(1)
            )
            counts[start:end] += 1

        counts = counts.clamp(min=1).unsqueeze(1)
        llr = (mutant_sum / counts) - (wildtype_sum.unsqueeze(1) / counts)

        # The wildtype-vs-wildtype entry is 0 by definition; zero it explicitly
        # so we can average exactly the 19 genuine substitutions.
        is_wt = mut_index.unsqueeze(0) == wt_ids.unsqueeze(1)
        llr = llr.masked_fill(is_wt, 0.0)
        sensitivity = llr.sum(dim=1) / (llr.shape[1] - is_wt.sum(dim=1)).clamp(min=1)

    return (
        sensitivity.detach().cpu().tolist(),
        llr.detach().cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# Per-residue detail derived from the LLR matrix
# ---------------------------------------------------------------------------
def verdict_for(score: float) -> str:
    if score < -8.0:
        return "Critical"
    if score < -4.0:
        return "Sensitive"
    return "Tolerant"


def residue_details(
    sequence: str, scores: Sequence[float], llr: Sequence[Sequence[float]]
) -> Dict[int, dict]:
    """Collapse the LLR matrix into what the hover card and table need.

    Keyed by 1-based residue number so it can be looked up directly by the
    `resi` field 3Dmol.js reports for a hovered atom.
    """
    details: Dict[int, dict] = {}
    for i, (wt, score, row) in enumerate(zip(sequence, scores, llr)):
        ranked = sorted(
            ((aa, val) for aa, val in zip(AMINO_ACIDS, row) if aa != wt),
            key=lambda pair: pair[1],
        )
        worst_aa, worst_val = ranked[0]
        best_aa, best_val = ranked[-1]
        details[i + 1] = {
            "wt": wt,
            "s": round(float(score), 2),
            "v": verdict_for(score),
            "worst": [worst_aa, round(float(worst_val), 2)],
            "best": [best_aa, round(float(best_val), 2)],
        }
    return details


# ---------------------------------------------------------------------------
# Structure painting
# ---------------------------------------------------------------------------
def paint_bfactors(pdb_text: str, scores: Sequence[float]) -> Tuple[str, int]:
    """Overwrite the PDB B-factor column with per-residue sensitivity scores.

    AlphaFold ships pLDDT in that column; swapping in our scores lets 3Dmol.js
    colour the ribbon with its own numeric gradient. Returns (pdb, n_painted).
    """
    painted = set()
    out_lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            try:
                resi = int(line[22:26])
            except ValueError:
                out_lines.append(line)
                continue
            idx = resi - 1  # AlphaFold numbers residues from 1
            if 0 <= idx < len(scores):
                value = max(-99.99, min(999.99, scores[idx]))
                line = f"{line[:60]}{value:6.2f}{line[66:]}"
                painted.add(resi)
            else:
                # Outside the scored window - park it at neutral white rather
                # than leaving a stale pLDDT that would render misleadingly blue.
                line = f"{line[:60]}{0.0:6.2f}{line[66:]}"
        out_lines.append(line)
    return "\n".join(out_lines), len(painted)


# py3Dmol funnels every argument through json.dumps, so a JS callback passed to
# setHoverable would arrive in the browser as a quoted *string* and never fire.
# We hand it these sentinels instead and splice the real functions in afterwards.
_HOVER_TOKEN = '"__DEEPMUTATE_HOVER__"'
_UNHOVER_TOKEN = '"__DEEPMUTATE_UNHOVER__"'

_HOVER_JS = """function(atom, viewer) {
    var d = window.DEEPMUTATE_DATA[atom.resi];
    if (!d) { return; }
    window.psShowPanel(atom.resi, d);
}"""

# setHoverable requires both callbacks. The panel is deliberately sticky - it
# keeps the last hovered residue on screen once the pointer moves away - so
# there is nothing to tear down here.
_UNHOVER_JS = """function(atom, viewer) {}"""

# The readout labels every number with the column heading it would carry in the
# sensitivity table, so a value read off the ribbon means the same thing as the
# value read off the row. The panel lives inside the iframe alongside the
# viewer, so hover updates need no cross-frame messaging.
_VIEWER_HELPERS_JS = r"""
window.PS_AA3 = {
    A:"Ala", C:"Cys", D:"Asp", E:"Glu", F:"Phe", G:"Gly", H:"His", I:"Ile",
    K:"Lys", L:"Leu", M:"Met", N:"Asn", P:"Pro", Q:"Gln", R:"Arg", S:"Ser",
    T:"Thr", V:"Val", W:"Trp", Y:"Tyr"
};

// Column headings, mirroring the sensitivity table.
window.PS_COLS = ["Position", "Wildtype", "Sensitivity (mean LLR)", "Verdict",
                  "Most damaging", "Most tolerated"];

window.psFields = function(resi, d) {
    var aa3 = window.PS_AA3[d.wt] || d.wt;
    var mut = function(pair) {
        return d.wt + resi + pair[0] + "  " + pair[1].toFixed(2);
    };
    return [
        String(resi),
        aa3 + " (" + d.wt + ")",
        d.s.toFixed(2),
        d.v,
        mut(d.worst),
        mut(d.best)
    ];
};

window.psShowPanel = function(resi, d) {
    var el = document.getElementById("ps-panel");
    if (!el) { return; }
    if (!d) {
        var blanks = "";
        for (var j = 0; j < window.PS_COLS.length; j++) {
            blanks += '<div class="ps-field"><span class="ps-lab">'
                    + window.PS_COLS[j] + '</span><span class="ps-val ps-dim">--</span></div>';
        }
        el.innerHTML = '<span class="ps-res ps-dim">--</span>' + blanks
                     + '<div class="ps-field"><span class="ps-lab">&nbsp;</span>'
                     + '<span class="ps-hint">Hover the ribbon</span></div>';
        return;
    }
    var tone = d.s < -8 ? "#f87171" : (d.s < -4 ? "#fbbf24" : "#60a5fa");
    var vals = window.psFields(resi, d);
    var html = '<span class="ps-res" style="color:' + tone + '">' + d.wt + resi + '</span>';
    for (var i = 0; i < window.PS_COLS.length; i++) {
        var style = (i === 2 || i === 3) ? ' style="color:' + tone + '"' : '';
        html += '<div class="ps-field"><span class="ps-lab">' + window.PS_COLS[i]
              + '</span><span class="ps-val"' + style + '>' + vals[i] + '</span></div>';
    }
    el.innerHTML = html;
};
"""


def render_structure(
    pdb_text: str, scores: Sequence[float], details: Dict[int, dict]
) -> str:
    """Build an interactive py3Dmol viewport with per-residue hover readouts."""
    painted_pdb, n_painted = paint_bfactors(pdb_text, scores)

    # Symmetric limits keep the midpoint of the red/white/blue gradient at 0.0,
    # so "white" always means "neutral mutation tolerance" for every protein.
    span = max(abs(min(scores)), abs(max(scores))) if scores else 1.0
    span = max(span, 1e-3)

    viewer = py3Dmol.view(width=820, height=520)
    viewer.addModel(painted_pdb, "pdb")
    viewer.setStyle(
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "rwb", # red = min (fragile), blue = max (tolerant)
                    "min": -span,
                    "max": span,
                }
            }
        }
    )
    viewer.setHoverDuration(80)
    viewer.setHoverable({}, True, "__DEEPMUTATE_HOVER__", "__DEEPMUTATE_UNHOVER__")
    viewer.setBackgroundColor("0x0f1419")
    viewer.zoomTo()

    viewer_html = viewer._make_html()
    viewer_html = viewer_html.replace(_HOVER_TOKEN, _HOVER_JS)
    viewer_html = viewer_html.replace(_UNHOVER_TOKEN, _UNHOVER_JS)

    data_json = json.dumps({str(k): v for k, v in details.items()})
    document = (
        "<style>"
        f"body{{margin:0;font-family:{FONT_STACK};background:#0f1419;color:#e5e7eb}}"
        "#ps-panel{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:flex-end;"
        "padding:9px 12px;background:#111827;border-bottom:1px solid #1f2937}"
        ".ps-field{display:flex;flex-direction:column;gap:2px}"
        ".ps-lab{font-size:9.5px;letter-spacing:.7px;text-transform:uppercase;"
        "color:#6b7280;white-space:nowrap}"
        f".ps-val{{font-family:{MONO_STACK};font-size:13px;color:#e5e7eb;"
        "white-space:nowrap}}"
        f".ps-res{{font-family:{MONO_STACK};font-size:19px;font-weight:700;"
        "line-height:1.15;padding-right:4px;align-self:flex-end}}"
        ".ps-dim{color:#4b5563}"
        ".ps-hint{font-size:12px;color:#6b7280;white-space:nowrap}"
        "#ps-legend{display:flex;align-items:center;gap:10px;padding:8px 12px;"
        f"font-size:12px;color:#9ca3af;background:#111827;"
        "border-top:1px solid #1f2937}"
        "</style>"
        "<div id='ps-panel'></div>"
        f"<script>window.DEEPMUTATE_DATA = {data_json};{_VIEWER_HELPERS_JS}"
        "window.psShowPanel(null, null);</script>"
        f"{viewer_html}"
        "<div id='ps-legend'>"
        "<span>Sensitivity (mean LLR)</span>"
        f"<span>fragile {-span:.2f}</span>"
        "<span style='flex:1;height:10px;border-radius:5px;"
        "background:linear-gradient(90deg,#d7191c,#ffffff,#2c7bb6)'></span>"
        f"<span>tolerant +{span:.2f}</span>"
        f"<span style='opacity:.75'>&middot; {n_painted} residues painted</span>"
        "</div>"
    )

    srcdoc = html_lib.escape(document, quote=True)
    return (
        f"<iframe srcdoc=\"{srcdoc}\" width='100%' height='620' "
        "style='border:1px solid #d1d5db;border-radius:8px;background:#0f1419' "
        "sandbox='allow-scripts allow-same-origin'></iframe>"
    )


def placeholder_panel(message: str) -> str:
    return (
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:520px;border:1px dashed #d1d5db;border-radius:8px;color:#6b7280;"
        f"font-family:{FONT_STACK};font-size:14px;line-height:1.5;"
        "text-align:center;padding:24px'>"
        f"{html_lib.escape(message)}</div>"
    )


# ---------------------------------------------------------------------------
# Downloadable exports
# ---------------------------------------------------------------------------
_EXPORT_DIR = pathlib.Path(tempfile.gettempdir()) / "deepmutate3d_exports"


def _export_path(uniprot_id: str, suffix: str) -> pathlib.Path:
    """A stable, human-readable filename per protein and artefact type."""
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (uniprot_id or "sequence").strip().upper() or "sequence"
    stem = re.sub(r"[^A-Za-z0-9_-]", "", stem)[:24] or "sequence"
    return _EXPORT_DIR / f"DeepMutate3D_{stem}_{suffix}"


def export_table(table: pd.DataFrame, uniprot_id: str) -> str:
    """Write the sensitivity table to CSV and return its path."""
    path = _export_path(uniprot_id, "sensitivity.csv")
    table.to_csv(path, index=False)
    return str(path)


def export_structure(
    pdb_text: str,
    scores: Sequence[float],
    uniprot_id: str,
    mode: str,
    model_id: str = MODEL_ID,
) -> str:
    """Write the scored structure to a PDB file and return its path.

    The B-factor column carries the Residue Sensitivity Score instead of
    AlphaFold's pLDDT, so the file opens in PyMOL or ChimeraX and can be coloured
    by B-factor directly - e.g. in PyMOL:  spectrum b, red_white_blue

    REMARK lines record that substitution, because a B-factor column silently
    meaning something else is exactly the kind of thing that misleads later.
    """
    painted, _ = paint_bfactors(pdb_text, scores)
    span = max(abs(min(scores)), abs(max(scores))) if scores else 0.0
    header = "\n".join(
        [
            "REMARK 300 GENERATED BY DEEPMUTATE-3D",
            "REMARK 300 THE B-FACTOR COLUMN HAS BEEN OVERWRITTEN. IT CONTAINS THE",
            "REMARK 300 PER-RESIDUE SENSITIVITY SCORE (MEAN LOG-LIKELIHOOD RATIO",
            "REMARK 300 OVER THE 19 NON-WILDTYPE SUBSTITUTIONS), *NOT* PLDDT.",
            f"REMARK 300 MODEL: {model_id}",
            f"REMARK 300 SCORING MODE: {mode.upper()}",
            f"REMARK 300 RANGE: -{span:.2f} (FRAGILE) TO +{span:.2f} (TOLERANT)",
            "REMARK 300 COORDINATES FROM ALPHAFOLD DB, CC-BY-4.0. CITE JUMPER 2021.",
            "REMARK 300 SUGGESTED PYMOL COLOURING: spectrum b, red_white_blue",
        ]
    )
    path = _export_path(uniprot_id, "scored.pdb")
    path.write_text(header + "\n" + painted + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Gradio orchestration
# ---------------------------------------------------------------------------
EXAMPLE_PROTEINS: Dict[str, str] = {
    "Haemoglobin subunit alpha, P69905 (142 aa)": "P69905",
    "Myoglobin, P02144 (154 aa)": "P02144",
    "Insulin, P01308 (110 aa)": "P01308",
    "Lysozyme C, P61626 (148 aa)": "P61626",
    "Ubiquitin, P0CG48 (685 aa)": "P0CG48",
    "Tumour suppressor p53, P04637 (393 aa)": "P04637",
    "SARS-CoV-2 spike glycoprotein, P0DTC2 (1273 aa)": "P0DTC2",
}


def load_example(choice: str):
    """Fetch the chosen protein's canonical sequence and fill both input boxes."""
    uniprot_id = EXAMPLE_PROTEINS.get(choice, "")
    if not uniprot_id:
        return gr.update(), gr.update(), "Pick a protein from the list first."
    sequence, note = fetch_uniprot_sequence(uniprot_id)
    if not sequence:
        return gr.update(), uniprot_id, f"Could not load example: {note}"
    label = choice.split(", ")[0]
    return (
        sequence,
        uniprot_id,
        f"Loaded **{label}** ({uniprot_id}, {len(sequence)} residues). "
        "Press *Scan Mutations & Render 3D Fold* to run it.",
    )


def build_table(
    sequence: str, scores: Sequence[float], details: Dict[int, dict]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pos": range(1, len(scores) + 1),
            "WT": list(sequence[: len(scores)]),
            "Sensitivity": [round(float(s), 3) for s in scores],
            "Verdict": [details[i + 1]["v"] for i in range(len(scores))],
            "Most damaging": [
                f"{details[i + 1]['wt']}{i + 1}{details[i + 1]['worst'][0]} "
                f"({details[i + 1]['worst'][1]:.2f})"
                for i in range(len(scores))
            ],
            "Most tolerated": [
                f"{details[i + 1]['wt']}{i + 1}{details[i + 1]['best'][0]} "
                f"({details[i + 1]['best'][1]:.2f})"
                for i in range(len(scores))
            ],
        }
    )


def scan(
    sequence_input: str,
    uniprot_input: str,
    mode: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    progress=gr.Progress(),
):
    """End-to-end handler wired to the submit button."""
    notes: List[str] = []
    sequence = clean_sequence(sequence_input)

    # Convenience: an empty sequence box with a UniProt ID pulls the canonical
    # sequence, which also guarantees the scores align with the AlphaFold model.
    if not sequence and uniprot_input.strip():
        progress(0.05, desc="Fetching sequence from UniProt")
        fetched, note = fetch_uniprot_sequence(uniprot_input)
        notes.append(note)
        if fetched:
            sequence = fetched

    error = validate_sequence(sequence)
    if error:
        return (
            f"### Cannot scan\n{error}\n\n"
            "Paste a protein sequence (FASTA is fine), supply a UniProt ID, "
            "or load one of the example proteins.",
            pd.DataFrame(),
            placeholder_panel("Nothing to render yet."),
            gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False),
        )

    if len(sequence) > ABSOLUTE_MAX_RESIDUES:
        return (
            f"### Sequence too long\n{len(sequence)} residues exceeds the "
            f"{ABSOLUTE_MAX_RESIDUES}-residue guard rail. Scan a domain of "
            "interest instead, or raise `ABSOLUTE_MAX_RESIDUES` in `app.py` if "
            "you are happy to wait.",
            pd.DataFrame(),
            placeholder_panel("Nothing to render yet."),
            gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False),
        )

    windows = sequence_windows(len(sequence))
    if len(windows) > 1:
        notes.append(
            f"Scored with {len(windows)} overlapping {WINDOW_RESIDUES}-residue "
            f"windows ({WINDOW_OVERLAP}-residue overlap) to cover the full "
            f"{len(sequence)}-residue sequence."
        )
    if not mode.startswith("Wildtype") and len(sequence) > 600:
        notes.append(
            "Long sequence in masked-marginal mode - switch to *Wildtype "
            "marginals (fast)* if you want a quicker first look."
        )

    model_id, _ = resolve_model(model_label)
    progress(
        0.15,
        desc=f"Scoring {len(sequence)} residues on {resolve_device()} "
        f"with {model_label}",
    )
    _t0 = time.perf_counter()
    scores, llr = compute_sensitivity(sequence, mode, model_label)
    scan_seconds = time.perf_counter() - _t0
    details = residue_details(sequence, scores, llr)
    table = build_table(sequence, scores, details)

    progress(0.8, desc="Retrieving AlphaFold structure")
    pdb_text, structure_note = fetch_alphafold_pdb(uniprot_input)
    notes.append(structure_note)

    csv_path = export_table(table, uniprot_input)
    csv_update = gr.update(value=csv_path, interactive=True)

    if pdb_text:
        progress(0.9, desc="Painting 3D heatmap")
        viewer_html = render_structure(pdb_text, scores, details)
        pdb_update = gr.update(
            value=export_structure(pdb_text, scores, uniprot_input, mode, model_id),
            interactive=True,
        )
    else:
        viewer_html = placeholder_panel(
            "No 3D structure loaded. The sensitivity scan is still valid.\n"
            + structure_note
        )
        pdb_update = gr.update(value=None, interactive=False)

    hottest = table.nsmallest(5, "Sensitivity")
    hotspot_text = ", ".join(
        f"{wt}{pos} ({score:.2f})"
        for pos, wt, score in zip(
            hottest["Pos"], hottest["WT"], hottest["Sensitivity"]
        )
    )
    summary = (
        "### Scan complete\n"
        f"- **Residues scored:** {len(sequence)} &nbsp;|&nbsp; "
        f"**Device:** `{resolve_device()}` &nbsp;|&nbsp; **Mode:** {mode}\n"
        f"- **Model:** `{model_id}` &nbsp;|&nbsp; "
        f"**Inference time:** {scan_seconds:.2f} s "
        f"({1000 * scan_seconds / max(len(sequence), 1):.1f} ms/residue)\n"
        f"- **Mean sensitivity:** {table['Sensitivity'].mean():.3f} "
        f"&nbsp;|&nbsp; **Most fragile:** {hotspot_text}\n"
        + "".join(f"- {n}\n" for n in notes)
    )
    return summary, table, viewer_html, csv_update, pdb_update


# Stage the weights now so ZeroGPU can optimise the CUDA placement at startup.
preload_model()


CSS = f"""
.deepmutate-title h1 {{ margin-bottom: 4px; }}
body, .gradio-container, .gradio-container * {{
    font-family: {FONT_STACK} !important;
}}
.gradio-container code, .gradio-container pre,
.gradio-container textarea, .gradio-container .cm-editor {{
    font-family: {MONO_STACK} !important;
}}
"""

# Gradio 6 moved `theme`/`css` from the Blocks constructor to launch(); older
# 4.x/5.x builds only accept them on Blocks. Detect once and route accordingly.
_THEME = gr.themes.Soft(
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
)
_LAUNCH_ACCEPTS_THEME = "theme" in inspect.signature(gr.Blocks.launch).parameters
LAUNCH_KWARGS = {"theme": _THEME, "css": CSS} if _LAUNCH_ACCEPTS_THEME else {}
BLOCKS_KWARGS = {} if _LAUNCH_ACCEPTS_THEME else {"theme": _THEME, "css": CSS}


with gr.Blocks(title="DeepMutate-3D", **BLOCKS_KWARGS) as demo:
    if LOGO_DATA_URI:
        gr.HTML(
            '<div style="display:flex;justify-content:center;padding:4px 0 2px">'
            f'<img src="{LOGO_DATA_URI}" alt="DeepMutate-3D" '
            'style="max-width:420px;width:100%;height:auto;border-radius:10px">'
            "</div>"
        )
    gr.Markdown(
        "# DeepMutate-3D\n"
        "**Protein-language-model mutation scanning, painted onto the AlphaFold fold.**\n\n"
        "ESM-2 scores every possible point mutation as a log-likelihood ratio against "
        "the wildtype residue. Positions the model refuses to change are evolutionarily "
        "constrained, likely structural or functional cores, and glow **red**. "
        "Positions it happily swaps glow **blue**.",
        elem_classes="deepmutate-title",
    )

    with gr.Row():
        example_dropdown = gr.Dropdown(
            choices=list(EXAMPLE_PROTEINS.keys()),
            value=list(EXAMPLE_PROTEINS.keys())[0],
            label="Example protein",
            scale=3,
        )
        load_button = gr.Button("Load sequence", variant="secondary", scale=1)

    with gr.Row():
        sequence_box = gr.Textbox(
            label="Protein sequence (FASTA or raw)",
            placeholder="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF...\n"
            "Leave blank to auto-fetch from the UniProt ID.",
            lines=8,
            scale=3,
        )
        with gr.Column(scale=1):
            uniprot_box = gr.Textbox(
                label="UniProt ID",
                placeholder="P69905",
                info="Used for the AlphaFold structure, and to fetch the sequence.",
            )
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value=DEFAULT_MODEL_LABEL,
                label="Model",
                info="Larger models score better but cost more compute.",
            )
            mode_radio = gr.Radio(
                choices=[
                    "Masked marginals (accurate)",
                    "Wildtype marginals (fast)",
                ],
                value="Masked marginals (accurate)",
                label="Scoring mode",
                info="Masked marginals runs one forward pass per residue.",
            )

    scan_button = gr.Button(
        "Scan Mutations & Render 3D Fold", variant="primary", size="lg"
    )

    with gr.Row():
        download_csv = gr.DownloadButton(
            "⬇ Sensitivity table (.csv)", size="sm", interactive=False, scale=1
        )
        download_pdb = gr.DownloadButton(
            "⬇ Scored structure (.pdb)", size="sm", interactive=False, scale=1
        )

    status_md = gr.Markdown("Awaiting input.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Residue Sensitivity Array")
            score_table = gr.Dataframe(
                headers=[
                    "Pos",
                    "WT",
                    "Sensitivity",
                    "Verdict",
                    "Most damaging",
                    "Most tolerated",
                ],
                interactive=False,
                max_height=620,
            )
        with gr.Column(scale=2):
            gr.Markdown("### Structural Heatmap (AlphaFold DB)")
            structure_html = gr.HTML(
                placeholder_panel("Run a scan to render the 3D fold.")
            )

    load_button.click(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[sequence_box, uniprot_box, status_md],
    )
    scan_button.click(
        fn=scan,
        inputs=[sequence_box, uniprot_box, mode_radio, model_dropdown],
        outputs=[status_md, score_table, structure_html, download_csv, download_pdb],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        share=False,
        **LAUNCH_KWARGS,
    )
