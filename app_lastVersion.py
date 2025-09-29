import os, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
import uuid

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import altair as alt
import shap
from PIL import Image
from io import BytesIO
import streamlit.components.v1 as components

# Import sklearn components for KB functionality
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- FP bit → substructure mapping (same radius/bits as training) ----
def fp_with_bitinfo(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    bitInfo = {}
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo)
    return mol, bitInfo, bv

def top_fp_bits_from_shap(shap_vector, top_k=6, bitInfo=None):
    """
    Return a list of (bit_idx, shap_val) for the top_k fingerprint bits.
    If bitInfo is provided (dict of mappable bits for the molecule), we only
    consider those bit indices so the returned bits are guaranteed mappable.
    """
    sv = np.asarray(shap_vector).ravel()
    if sv.size == 0:
        return []

    # limit sv to fingerprint region
    fp_sv = sv[:FP_BITS] if sv.size >= FP_BITS else sv

    if bitInfo:
        # bitInfo keys may be ints or strings; normalize to ints
        try:
            mappable_keys = set(int(k) for k in bitInfo.keys())
        except Exception:
            mappable_keys = set()

        # build list of (bit_idx, shap_val) only for keys present in bitInfo
        candidates = []
        for k in mappable_keys:
            if k < fp_sv.size:
                val = float(fp_sv[k])
            else:
                val = 0.0
            if val != 0.0:
                candidates.append((k, val))

        # sort by abs(shap) desc and return top_k
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        return candidates[:top_k]
    else:
        # fallback: original behaviour (returns tuples)
        idx = np.argsort(np.abs(fp_sv))[::-1]
        return [(int(i), float(fp_sv[i])) for i in idx[:top_k] if fp_sv[i] != 0.0]


def atoms_bonds_for_bits(mol, bitInfo, bits, radius=2):
    """
    bits can be:
      - iterable of (bit_idx, shap_val) tuples (preferred), or
      - iterable of int bit_idx (then shap_val is assumed 1.0)
    Returns (atoms_set, bonds_set, atom_weight_dict)
    """
    atoms, bonds, atom_weight = set(), set(), {}
    for item in bits:
        # normalize to (bit_idx, shap_val)
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            try:
                bit_idx = int(item[0])
                shap_val = float(item[1]) if len(item) > 1 else 1.0
            except Exception:
                continue
        else:
            try:
                bit_idx = int(item)
                shap_val = 1.0
            except Exception:
                continue

        if bit_idx not in bitInfo:
            continue

        envs = bitInfo[bit_idx]  # list of (centerAtomIdx, radius)
        for center, rad in envs:
            bond_idxs = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
            atom_idxs = set([center] +
                            [mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in bond_idxs] +
                            [mol.GetBondWithIdx(b).GetEndAtomIdx() for b in bond_idxs])
            atoms.update(atom_idxs); bonds.update(bond_idxs)
            for a in atom_idxs:
                atom_weight[a] = max(atom_weight.get(a, 0.0), abs(shap_val))
    return atoms, bonds, atom_weight

def render_copy_button(text: str, key: str):
    # Simplified copy button with better error handling and reduced JS complexity
    safe_text = json.dumps(text)
    button_id = f"btn_{key}_{hash(text) % 10000}"  # Shorter, more reliable ID
    
    html = f"""
    <div style="display:flex;align-items:center;gap:8px;">
      <button id="{button_id}" onclick="copyText()" style="padding:4px 8px;border-radius:4px;border:1px solid #ddd;background:#f8f9fa;cursor:pointer;font-size:12px;">
        Copy
      </button>
    </div>
    <script>
    function copyText() {{
      const btn = document.getElementById("{button_id}");
      if (!btn) return;
      
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText({safe_text}).then(function() {{
            btn.innerText = "✓";
            setTimeout(function() {{ btn.innerText = "Copy"; }}, 1000);
          }}).catch(function() {{ 
            fallbackCopy();
          }});
        }} else {{
          fallbackCopy();
        }}
      }} catch (e) {{
        fallbackCopy();
      }}
      
      function fallbackCopy() {{
        const textArea = document.createElement("textarea");
        textArea.value = {safe_text};
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.select();
        try {{
          document.execCommand('copy');
          btn.innerText = "✓";
          setTimeout(function() {{ btn.innerText = "Copy"; }}, 1000);
        }} catch (e) {{
          btn.innerText = "Error";
          setTimeout(function() {{ btn.innerText = "Copy"; }}, 1500);
        }}
        document.body.removeChild(textArea);
      }}
    }}
    </script>
    """
    # Reduced height to minimize DOM impact
    components.html(html, height=30)

def draw_payload_with_shap(smiles, shap_vector, top_k=6, radius=2):
    """
    Returns a payload dict (or a payload-like dict with error/note fields).
    Field summary:
      - smiles, atoms, bonds, atom_scores, had_highlights (bool)
      - top_bits: list of (bit_idx, shap_val) used
      - top_bits_err, fallback_info, bitInfo_keys_preview for debugging
      - If an unrecoverable error occurs, returns {"error": "..."}
    """
    # 0) Basic sanitization
    if smiles is None:
        return {"error": "No SMILES provided."}
    smiles_in = str(smiles).strip()
    if smiles_in == "":
        return {"error": "SMILES is empty after stripping whitespace."}

    # 1) RDKit parse / canonicalize
    try:
        mol_check = Chem.MolFromSmiles(smiles_in)
        if mol_check is None:
            return {"error": f"RDKit cannot parse SMILES: {repr(smiles_in)}"}
        smiles_canon = Chem.MolToSmiles(mol_check)
    except Exception as e:
        return {"error": f"RDKit parse error: {e}"}

    # 2) fingerprint + bitInfo
    try:
        mol, bitInfo, _ = fp_with_bitinfo(smiles_canon, nBits=FP_BITS, radius=radius)
        if mol is None:
            return {"error": "fp_with_bitinfo returned mol=None (fingerprint failure)."}
    except Exception as e:
        return {"error": f"fp_with_bitinfo exception: {e}"}

    # small debug preview of bitInfo keys
    try:
        bitinfo_keys_preview = list(bitInfo.keys())[:20] if isinstance(bitInfo, dict) else None
    except Exception:
        bitinfo_keys_preview = None

    # 3) Compute top bits (prefer bits that actually appear in bitInfo)
    top_bits = []
    top_bits_err = None
    fallback_info = None

    try:
        raw_top_bits = top_fp_bits_from_shap(shap_vector, top_k=top_k, bitInfo=bitInfo)
    except Exception as e:
        raw_top_bits = []
        top_bits_err = f"top_fp_bits_from_shap raised: {e}"

    # If raw_top_bits is falsy, pick top_k bits from bitInfo by mapping size (guaranteed mappable)
    if not raw_top_bits:
        try:
            key_scores = []
            if isinstance(bitInfo, dict):
                for k, v in bitInfo.items():
                    try:
                        key_scores.append((int(k), len(v) if v is not None else 0))
                    except Exception:
                        continue
            key_scores.sort(key=lambda x: x[1], reverse=True)
            # produce list of (bit_idx, 0.0) tuples for consistency with atoms_bonds_for_bits
            raw_top_bits = [(int(k), 0.0) for k, _ in key_scores[:top_k]]
            fallback_info = f"Fallback selected {len(raw_top_bits)} bits from bitInfo by env size."
        except Exception as e:
            raw_top_bits = []
            fallback_info = f"Fallback failed: {e}"

    # Normalize raw_top_bits: ensure list of (int_bit_idx, float_shap_val)
    normalized_top_bits = []
    for item in raw_top_bits:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                bit_idx = int(item[0])
                shap_val = float(item[1]) if len(item) > 1 else 0.0
            else:
                bit_idx = int(item)
                shap_val = 0.0
            normalized_top_bits.append((bit_idx, shap_val))
        except Exception:
            continue

    top_bits = normalized_top_bits

    # 4) Map bits -> atoms/bonds and compute atom weights
    try:
        atoms, bonds, atom_weight = atoms_bonds_for_bits(mol, bitInfo, top_bits, radius=radius)
    except Exception as e:
        return {
            "error": f"atoms_bonds_for_bits raised exception: {e}",
            "top_bits": top_bits,
            "top_bits_err": top_bits_err,
            "fallback_info": fallback_info,
            "bitInfo_keys_preview": bitinfo_keys_preview
        }

    # 5) If no atoms found, return payload with diagnostic info (still include canonical SMILES)
    if not atoms:
        return {
            "smiles": smiles_canon,
            "atoms": [],
            "bonds": [],
            "atom_scores": {},
            "had_highlights": False,
            "note": "No atoms mapped for selected fingerprint bits. Try increasing top_k or check SHAP vector.",
            "top_bits": top_bits,
            "top_bits_err": top_bits_err,
            "fallback_info": fallback_info,
            "bitInfo_keys_preview": bitinfo_keys_preview
        }

    # 6) Build normalized scores
    wmax = max(atom_weight.values()) if atom_weight else 1.0
    atoms = [int(a) for a in atoms]
    bonds = [int(b) for b in bonds]
    scores = {int(a): float(atom_weight.get(a, 0.0) / wmax) for a in atoms}

    # final payload
    return {
        "smiles": smiles_canon,
        "atoms": atoms,
        "bonds": bonds,
        "atom_scores": scores,
        "had_highlights": True,
        "top_bits": top_bits,
        "top_bits_err": top_bits_err,
        "fallback_info": fallback_info,
        "bitInfo_keys_preview": bitinfo_keys_preview
    }

# ---------- Heuristic Copilot (no LLM) ----------

def compute_descriptors(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HAcc": Descriptors.NumHAcceptors(mol),
        "HDon": Descriptors.NumHDonors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "Rings": Descriptors.RingCount(mol),
    }

def rank_top_risks(mt_probs: dict, top_k=3):
    return sorted(mt_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

def heuristic_recs(desc, mt_probs, thr=0.5):
    recs = []
    
    # Safety check for empty or invalid desc dictionary
    if not desc or not isinstance(desc, dict):
        return ["Unable to generate recommendations - missing molecular descriptors."]
    
    # Physchem heuristics (simple, explainable) with safe key access
    logp = desc.get("LogP", 0.0)
    tpsa = desc.get("TPSA", 0.0)
    molwt = desc.get("MolWt", 0.0)
    hacc = desc.get("HAcc", 0)
    hdon = desc.get("HDon", 0)
    
    if logp > 3.5:
        recs.append("High LogP—reduce lipophilicity (add heteroatoms or trim alkyl bulk).")
    if tpsa < 20:
        recs.append("Low TPSA—add polar groups to balance permeability vs toxicity.")
    if molwt > 450:
        recs.append("High MW—simplify scaffold or remove heavy substituents.")
    if hacc + hdon < 2 and logp > 3.0:
        recs.append("Very apolar—introduce H-bonding functionality.")
    
    # Pathway-specific hints
    if mt_probs and isinstance(mt_probs, dict):
        high = {k:v for k,v in mt_probs.items() if v >= thr}
        if any(k.startswith("NR-ER") for k in high):
            recs.append("ER risk—avoid phenols/steroid-mimetic hydrophobics; consider polarity increase.")
        if any(k.startswith("NR-AR") for k in high):
            recs.append("AR risk—de-emphasize steroid-like cores; add polarity or break planarity.")
        if any(k.startswith("SR-ARE") for k in high):
            recs.append("Oxidative stress—mitigate reactive motifs (nitro/anilines); use bioisosteres.")
        if any(k.startswith("SR-HSE") for k in high):
            recs.append("Heat-shock response—watch electrophiles; cap/replace potential warheads.")
    
    return recs

def copilot_summary_text(smiles, bin_proba, mt_probs, desc, top_k=3, thr=0.5):
    # Headline risk from binary probability
    headline = ("HIGH" if bin_proba >= thr else "MODERATE" if bin_proba >= (thr-0.15) else "LOW")
    top_paths = rank_top_risks(mt_probs, top_k=top_k)
    top_txt = ", ".join([f"{k}={v:.2f}" for k,v in top_paths]) if top_paths else "None"
    d = desc
    props = f"MW={d['MolWt']:.0f}, LogP={d['LogP']:.2f}, TPSA={d['TPSA']:.0f}, HBA={d['HAcc']}, HBD={d['HDon']}, Rings={d['Rings']}"
    recs = heuristic_recs(desc, mt_probs, thr=thr) or ["No specific red flags; review metabolism & soft spots."]
    bullets = "\n".join([f"- {r}" for r in recs])
    return (
        f"**AI Medic Summary (Heuristic)**\n"
        f"- Overall toxicity risk: **{headline}** (binary prob={bin_proba:.2f})\n"
        f"- Top pathways: {top_txt}\n"
        f"- Properties: {props}\n\n"
        f"**Suggestions**\n{bullets}\n"
    )

# -------- Copilot (refined, no duplicate top-pathways table) --------

def nice_bool(flag): 
    return "✅" if flag else "⚠️"

def copilot_bullets(desc: dict, mt_probs: dict, bin_proba: float, bin_thr: float = 0.5) -> list[str]:
    bullets = []

    # Overall stance
    if bin_proba >= bin_thr:
        bullets.append(f"**Overall risk:** {nice_bool(False)} Predicted **HIGH** toxicity risk (p={bin_proba:.2f}).")
    elif bin_proba >= (bin_thr - 0.15):
        bullets.append(f"**Overall risk:** {nice_bool(True)} **Moderate** risk (p={bin_proba:.2f}); warrants structure tweaks.")
    else:
        bullets.append(f"**Overall risk:** {nice_bool(True)} **Low** risk (p={bin_proba:.2f}); still assess ADME/soft spots.")

    # Simple property readout heuristics
    if desc["LogP"] > 3.5:
        bullets.append("**Lipophilicity:** On the high side → consider trimming hydrophobics or adding heteroatoms.")
    elif desc["LogP"] < 1.0:
        bullets.append("**Lipophilicity:** Quite low → permeability/brain exposure may be limited; balance with hydrophobics.")

    if desc["TPSA"] < 20:
        bullets.append("**Polarity:** Very low TPSA → watch nonspecific binding; add polar handles if needed.")
    elif desc["TPSA"] > 90:
        bullets.append("**Polarity:** High TPSA → good solubility, but passive permeability may drop.")

    if desc["MolWt"] > 450:
        bullets.append("**Size:** Heavy scaffold → simplify or remove bulky substituents to improve developability.")
    elif desc["MolWt"] < 200:
        bullets.append("**Size:** Small core → favorable; ensure potency and selectivity aren’t compromised.")

    hb = desc["HAcc"] + desc["HDon"]
    if hb < 2 and desc["LogP"] > 3.0:
        bullets.append("**H-bonding:** Very apolar profile → introduce H-bond donors/acceptors to tune PK and reduce risk.")

    # Pathway-flavored guidance (no table)
    high = {k:v for k,v in mt_probs.items() if v >= 0.5}
    if any(k.startswith("SR-ARE") for k in high):
        bullets.append("**Oxidative stress (ARE):** Avoid reactive motifs (nitro/anilines); consider safer bioisosteres.")
    if any(k.startswith("NR-ER") for k in high):
        bullets.append("**ER signaling:** Steroid-mimetic features suspected → break planarity or increase polarity.")
    if any(k.startswith("NR-AR") for k in high):
        bullets.append("**AR signaling:** Reduce bulky hydrophobics; de-bias steroid-like cores.")

    # Lightweight toxicophore hints (optional SMARTS)
    smi = st.session_state.get("smiles", "")
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol:
        nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
        aniline = Chem.MolFromSmarts("c[NH2]")
        if mol.HasSubstructMatch(nitro):
            bullets.append("**Substructure alert:** Nitro group detected → common tox liability; consider replacement.")
        if mol.HasSubstructMatch(aniline):
            bullets.append("**Substructure alert:** Aniline moiety detected → watch for bioactivation/tox; explore isosteres.")

    # Closing nudge
    bullets.append("**Next step:** Iterate small edits (↓LogP, +polarity) and re-score to converge on a safer analog.")
    return bullets


def render_shap_rdkitjs(payload: dict, size=(600, 420)):
    width, height = size
    cid = f"rdkit_{hash(str(payload)) % 10000}"  # Shorter ID for better performance

    smiles_js = json.dumps(payload["smiles"])
    atoms_js  = json.dumps([int(a) for a in payload.get("atoms", [])])
    bonds_js  = json.dumps([int(b) for b in payload.get("bonds", [])])
    highlight_color_js = json.dumps([1.0, 0.6, 0.2])

    html = f"""
    <div id="{cid}" style="width:{width}px;height:{height}px;border:1px solid #eee;background:#f9f9f9;">
      <div style="padding:20px;text-align:center;color:#666;">Loading molecule visualization...</div>
    </div>
    <script>
    (function() {{
      const target = document.getElementById("{cid}");
      if (!target) return;
      
      let rdkitLoaded = false;
      let loadTimeout;
      
      function showError(msg) {{
        target.innerHTML = '<div style="padding:20px;font:12px sans-serif;color:#666;text-align:center;">' + msg + '</div>';
      }}
      
      function cleanup() {{
        if (loadTimeout) clearTimeout(loadTimeout);
      }}
      
      // Set timeout for loading
      loadTimeout = setTimeout(function() {{
        showError("Molecule rendering timed out. Try refreshing.");
        cleanup();
      }}, 8000);
      
      async function loadRDKit() {{
        if (window.RDKit) {{
          rdkitLoaded = true;
          return window.RDKit;
        }}
        
        if (window.initRDKitModule) {{
          try {{
            const rdkit = await window.initRDKitModule({{
              locateFile: (f) => "https://unpkg.com/@rdkit/rdkit/dist/" + f
            }});
            window.RDKit = rdkit;
            rdkitLoaded = true;
            return rdkit;
          }} catch (e) {{
            throw new Error("Failed to initialize RDKit: " + e.message);
          }}
        }}
        
        // Load script
        return new Promise((resolve, reject) => {{
          const script = document.createElement("script");
          script.src = "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js";
          script.onload = async function() {{
            try {{
              const rdkit = await initRDKitModule({{
                locateFile: (f) => "https://unpkg.com/@rdkit/rdkit/dist/" + f
              }});
              window.RDKit = rdkit;
              rdkitLoaded = true;
              resolve(rdkit);
            }} catch (e) {{
              reject(e);
            }}
          }};
          script.onerror = () => reject(new Error("Failed to load RDKit script"));
          document.head.appendChild(script);
        }});
      }}
      
      async function renderMolecule() {{
        try {{
          const RDKit = await loadRDKit();
          clearTimeout(loadTimeout);
          
          const smiles = {smiles_js};
          const atoms = {atoms_js};
          const bonds = {bonds_js};
          
          let mol;
          try {{
            mol = RDKit.get_mol(smiles);
            if (!mol) throw new Error("Invalid molecule");
          }} catch (e) {{
            showError("Invalid SMILES: " + smiles);
            return;
          }}
          
          const details = {{
            width: {width},
            height: {height},
            atoms: atoms,
            bonds: bonds,
            highlightColour: {highlight_color_js},
            fillHighlights: true,
            continuousHighlight: true,
            bondLineWidth: 1,
            kekulize: true,
            backgroundColour: [1,1,1,0]
          }};
          
          try {{
            const svg = mol.get_svg_with_highlights(JSON.stringify(details));
            target.innerHTML = svg;
          }} catch (e) {{
            showError("Rendering failed: " + e.message);
          }} finally {{
            if (mol) mol.delete();  // Memory cleanup
          }}
          
        }} catch (err) {{
          clearTimeout(loadTimeout);
          showError("Error: " + (err.message || "Unknown error"));
        }}
      }}
      
      // Start rendering
      renderMolecule();
      
      // Cleanup on page unload
      window.addEventListener('beforeunload', cleanup);
    }})();
    </script>
    """
    st.components.v1.html(html, height=height + 10)



# Using export_nb3_admet
ADMET_DIR = "export_nb3_admet"

@st.cache_resource
def load_admet_models():
    manifest_path = os.path.join(ADMET_DIR, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    models = {}
    meta = {}
    for d in manifest["datasets"]:
        name = d["name"]            # "esol" | "lipo" | "bbbp"
        path = os.path.join(ADMET_DIR, d["model_file"])
        models[name] = joblib.load(path)
        meta[name] = {"type": d["type"], "task": d["task"]}
    feat_cfg = manifest["feature_config"]
    return models, meta, feat_cfg

admet_models, admet_meta, admet_featcfg = load_admet_models()


def predict_admet(smiles: str):
    """Returns dict: key -> {'value': float, 'type': 'regression'|'classification'}"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = featurize_fp_desc(mol)  # FP+5 desc (2053)
    out = {}
    for name, model in admet_models.items():
        mtype = admet_meta[name]["type"]
        if mtype == "regression":
            val = float(model.predict([x])[0])
        else:
            val = float(model.predict_proba([x])[0, 1])  # probability BBBP=1
        out[name] = {"value": val, "type": mtype, "task": admet_meta[name]["task"]}
    return out

# Traffic-light interpreters (simple, transparent)
def interpret_logS(logS):
    # ESOL heuristic: > -3 good, -4..-3 moderate, < -4 poor
    if logS > -3:  return "✅ good"
    if logS > -4:  return "🟡 moderate"
    return "🔴 poor"

def interpret_logD(logD):
    # Lipophilicity sweet spot ~1–3 for many oral drugs
    if 1.0 <= logD <= 3.0: return "✅ in-range"
    if 0.5 <= logD < 1.0 or 3.0 < logD <= 3.5: return "🟡 borderline"
    return "🔴 out-of-range"

def interpret_bbbp(p, project="Non-CNS"):
    # Non-CNS: lower BBBP preferred; CNS: higher BBBP preferred
    if project == "CNS":
        if p >= 0.7: return "✅ high (desired)"
        if p >= 0.4: return "🟡 medium"
        return "🔴 low"
    else:
        if p <= 0.3: return "✅ low (desired)"
        if p <= 0.6: return "🟡 medium"
        return "🔴 high"

# Simple Developability Score (0..1). Tune weights per project.
def developability_score(logS, logD, bbbp_prob, project="Non-CNS"):
    # Normalize solubility: map [-6..0] → [0..1]
    sol_score = np.clip((logS + 6.0) / 6.0, 0.0, 1.0)
    # Lipophilicity score: peak at 2.0; radius 3.0 to zero
    lipo_score = 1.0 - np.clip(abs(logD - 2.0) / 3.0, 0.0, 1.0)
    # BBBP preference flips by project
    bbbp_score = bbbp_prob if project == "CNS" else (1.0 - bbbp_prob)

    # Weights: sol 0.4, lipo 0.4, BBBP 0.2
    score = 0.4 * sol_score + 0.4 * lipo_score + 0.2 * bbbp_score
    return float(np.clip(score, 0.0, 1.0))

# Wraps your existing pipeline (binary, multi-task, ADMET, descriptors, developability) and returns a clean dict the chatbot or any UI can consume.
def predict_all_models(smiles: str):
    """
    Run the full pipeline on one SMILES:
      - RDKit mol parse
      - descriptors (compute_descriptors)
      - FP+desc features for models
      - binary model prob + label (uses st.session_state.bin_thr)
      - multi-task probs (MT_TASKS / mt_models)
      - ADMET predictions (predict_admet)
      - developability composite (from admet preds)
    Returns a dict with consistent keys.
    """
    # 1) Parse
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES provided to predict_all_models()")

    # 2) Descriptors
    desc = compute_descriptors(mol)

    # 3) Binary prediction
    x_feat = featurize_fp_desc(mol)  # 2053 vector
    bin_proba = float(binary_model.predict_proba([x_feat])[0, 1])
    bin_thr = float(st.session_state.get("bin_thr", 0.5))
    bin_label = int(bin_proba >= bin_thr)

    # 4) Multi-task predictions
    mt_probs = {}
    for task in MT_TASKS:
        mdl = mt_models[task]
        try:
            mt_probs[task] = float(mdl.predict_proba([x_feat])[0, 1])
        except Exception:
            # fallback if model expects different input
            mt_probs[task] = float(mdl.predict([x_feat])[0] if hasattr(mdl, "predict") else np.nan)

    # 5) ADMET predictions (uses your predict_admet)
    admet_preds = predict_admet(smiles) or {}

    # 6) Compose developability score from ADMET preds (use same logic as UI)
    raw_logS = admet_preds.get("esol", {}).get("value", np.nan)
    raw_logD = admet_preds.get("lipo", {}).get("value", np.nan)
    raw_bbbp = admet_preds.get("bbbp", {}).get("value", np.nan)
    # Use your developability_score helper; default project=Non-CNS
    try:
        dev_score = developability_score(
            raw_logS if not np.isnan(raw_logS) else -6.0,
            raw_logD if not np.isnan(raw_logD) else 5.0,
            raw_bbbp if not np.isnan(raw_bbbp) else 0.5,
            project="Non-CNS"
        )
    except Exception:
        dev_score = None

    # 7) Compute lightweight SHAP for explainability (top descriptors only)
    try:
        top_shap_features = compute_lightweight_shap(mol, x_feat)
    except Exception:
        top_shap_features = ["MolWt", "LogP", "TPSA"]  # fallback

    # 8) (Optional) light SHAP summary: reuse existing desc_table in session if computed,
    # otherwise provide empty list. (Computing SHAP inside chat can be heavy; keep optional.)
    shap_summary = None
    if st.session_state.get("desc_table") is not None:
        try:
            # convert to list of "feature (shap)" strings
            df = st.session_state["desc_table"]
            shap_summary = [f"{r['feature']}({r['shap']:.3f})" for _, r in df.iterrows()]
        except Exception:
            shap_summary = None

    # 9) Package results
    return {
        "smiles": Chem.MolToSmiles(mol),
        "mol": mol,
        "descriptors": desc,
        "binary": {"prob": bin_proba, "label": bin_label, "threshold": bin_thr},
        "multitask": mt_probs,
        "admet": admet_preds,
        "developability": dev_score,
        "shap_summary": shap_summary,
        "top_shap_features": top_shap_features,  # New: molecule-specific top features
        "x_feat": x_feat  # raw feature vector for downstream uses (optional)
    }


def compute_lightweight_shap(mol, x_feat):
    """
    Compute SHAP values for descriptors only (most interpretable features)
    Returns top 3 descriptor names without numerical values for explainability display
    """
    try:
        import shap
        
        # Use binary model for SHAP computation
        X_sample = np.asarray(x_feat, dtype=float).reshape(1, -1)
        
        # TreeExplainer for binary model
        explainer = shap.TreeExplainer(binary_model)
        sv = explainer.shap_values(X_sample)
        
        # Handle multiclass output if needed
        if isinstance(sv, list):
            sv = sv[1]  # positive class
        
        shap_row = sv[0]
        
        # Focus on descriptor features (last 5 features in the vector)
        desc_names = ["MolWt", "LogP", "HAcc", "HDon", "TPSA"]
        desc_shap = shap_row[-5:]  # last 5 elements are descriptors
        
        # Create dataframe and sort by absolute SHAP values
        df = pd.DataFrame({
            "feature": desc_names,
            "shap": desc_shap
        })
        df["abs_shap"] = df["shap"].abs()
        df = df.sort_values("abs_shap", ascending=False)
        
        # Return top 3 feature names only (no numerical values)
        top_features = df.head(3)["feature"].tolist()
        return top_features
        
    except Exception as e:
        # Fallback to default descriptors if SHAP computation fails
        return ["MolWt", "LogP", "TPSA"]




# -----------------------------
# Memory Management & Session Cleanup
# -----------------------------

def cleanup_session_state():
    """Clean up heavy objects from session state to free memory"""
    cleanup_keys = [
        "explain_img", "explain_payload", "mol", "x_mt", 
        "desc_table", "last_heavy_computation"
    ]
    for key in cleanup_keys:
        if key in st.session_state:
            del st.session_state[key]

def reset_for_new_molecule():
    """Reset session state when analyzing a new molecule"""
    reset_keys = [
        "binary_proba", "binary_label", "mt_probs", "computed_desc",
        "developability_score", "explain_img", "desc_table", "explain_payload",
        "prediction_success_message", "canonicalization_message",
        "smiles", "smiles_input", "prediction_completed", "detailed_predict_clicked", "regular_predict_clicked",
        "scroll_molecule_done", "scroll_to_molecule_requested"  # Clear scroll tracking
    ]
    for key in reset_keys:
        if key in st.session_state:
            st.session_state[key] = None

# -----------------------------
# Paths 
# -----------------------------
BINARY_MODEL_PATH = "tox21_model_v1.pkl"          # binary toxic vs non-toxic
MULTITASK_EXPORT_DIR = "export_nb2_streamlit"     # contains manifest.json + one joblib per task

# -----------------------------
# Featurizers (must match training exactly)
# -----------------------------
FP_BITS = 2048
FP_RADIUS = 2
DESC5 = ["MolWt","MolLogP","NumHAcceptors","NumHDonors","TPSA"]

def featurize_fp(mol) -> np.ndarray:
    """2048-bit Morgan FP (ECFP4)."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
    return np.array(fp, dtype=float)

def featurize_fp_desc(mol) -> np.ndarray:
    """2048-bit FP + 5 descriptors (order fixed)."""
    fp_arr = featurize_fp(mol)
    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
    ], dtype=float)
    return np.concatenate([fp_arr, desc])

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource(show_spinner="Loading binary model...")
def load_binary_model():
    """Load binary model with memory optimization"""
    try:
        model = joblib.load(BINARY_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load binary model: {e}")
        return None

@st.cache_resource(show_spinner="Loading multi-task models...")
def load_multitask_models():
    """Load multi-task models with memory optimization"""
    try:
        manifest_path = os.path.join(MULTITASK_EXPORT_DIR, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Basic sanity about features (expect FP+desc)
        fc = manifest.get("feature_config", {})
        exp_feats = fc.get("featurization", "fp+descriptors")
        if exp_feats.lower() not in ["fp+descriptors", "fp_desc", "fp+desc"]:
            st.warning(f"Manifest featurization reported as '{exp_feats}'. "
                    "This app assumes FP+5 descriptors for multi-task models.")

        models = {}
        tasks = []
        for item in manifest["tasks"]:
            t = item["task"]
            p = os.path.join(MULTITASK_EXPORT_DIR, item["model_file"])
            try:
                models[t] = joblib.load(p)
                tasks.append(t)
            except Exception as e:
                st.warning(f"Failed to load model for task {t}: {e}")
        
        return models, tasks, fc
    except Exception as e:
        st.error(f"Failed to load multi-task models: {e}")
        return {}, [], {}

binary_model = load_binary_model()
mt_models, MT_TASKS, mt_feat_cfg = load_multitask_models()

# Validation that models loaded successfully
if binary_model is None:
    st.error("⚠️ Binary model failed to load. Some features may not work.")
if not mt_models:
    st.error("⚠️ Multi-task models failed to load. Some features may not work.")
elif len(mt_models) < len(MT_TASKS):
    st.warning(f"⚠️ Only {len(mt_models)}/{len(MT_TASKS)} multi-task models loaded successfully.")


import time
import datetime
import streamlit as st
import textwrap
import json
import numpy as np
import os

def medic_copilot_ui(immediate_analyze=False):
    """
    Polished Copilot UI.
    - Uses st.chat_message when available, otherwise column-based bubbles.
    - Renders only minimal controls when not analyzed; after analyze, advanced tabs can show.
    - Expects predict_all_models() or st.session_state already populated after analysis.
    """
    # SPACING REDUCTION - Move copilot up with reduced spacing by 50% more
    st.markdown('<div style="margin-top: -45px; margin-bottom: -20px;"></div>', unsafe_allow_html=True)
    
    # open wrapper for Copilot (so we can hide the whole block later)
    st.markdown('<div id="copilot_container">', unsafe_allow_html=True)

    def build_copilot_card():
        import numpy as np
        
        # Safe getter function that handles None values
        def safe_float(key, default=0.0):
            value = st.session_state.get(key, default)
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_dict(key, default=None):
            value = st.session_state.get(key, default)
            return value if value is not None else (default or {})
        
        ctx = {
            "binary_proba": safe_float("binary_proba", 0.0),
            "bin_thr": safe_float("bin_thr", 0.5),
            "mt_probs": safe_dict("mt_probs", {}),
            "developability": safe_float("developability_score", None),
            "desc": safe_dict("computed_desc", {}),
            "shap_summary": shap_short_phrases(st.session_state.get("desc_table", None), topn=4) if st.session_state.get("copilot_use_shap", True) else [],
            "heuristic_recs": heuristic_recs(safe_dict("computed_desc", {}), safe_dict("mt_probs", {}), thr=safe_float("bin_thr", 0.5)) if "heuristic_recs" in globals() else []
        }
        # Try to load knowledge base with enhanced error handling and memory protection
        # Knowledge Base Integration (optimized system)
        kb_res = []
        if ENABLE_KB and kb_index is not None:
            try:
                kb_query = " ".join(ctx["shap_summary"])
                if kb_query.strip():
                    kb_res = query_knowledge_base(kb_query, top_k=1)
            except Exception as e:
                st.warning(f"Knowledge base query failed: {e}")
                kb_res = []
            
        ctx["kb_snips"] = kb_res

        bin_prob = float(ctx.get("binary_proba", 0.0))
        bin_thr = float(ctx.get("bin_thr", 0.5))
        dev_score = ctx.get("developability", None)
        desc = ctx.get("desc", {}) or {}
        mt = ctx.get("mt_probs", {}) or {}
        heur = ctx.get("heuristic_recs", []) or []
        shap_summary = ctx.get("shap_summary", []) or []
        # Use kb_res directly in the template
        
        def tox_badge_html(p, thr):
            color = "#059669" if p < thr else "#b91c1c"
            text = f"{p:.2f}"
            return f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:{color};color:white;font-weight:700'>{text}</div>"
        
        # Format developability score safely
        dev_display = f"{dev_score:.2f}" if dev_score is not None else "N/A"
        
        header_html = f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px">
        <div style="font-size:18px;font-weight:800;color:#0f172a">MolGenie — AI-powered molecular insights</div>
        <div style="display:flex;align-items:center;gap:8px">
            <div style="font-size:12px;color:#0f172a">Tox prob</div>
            {tox_badge_html(bin_prob, bin_thr)}
            {"<div style='width:8px'></div>"}
            <div style="font-size:12px;color:#0f172a;margin-left:6px">Developability</div>
            <div style='display:inline-block;padding:6px 10px;border-radius:8px;background:#eef2ff;color:#084298;font-weight:600'>
            {dev_display}
            </div>
        </div>
        </div>
        """
        molwt = f"{desc.get('MolWt', np.nan):.0f}" if desc else "—"
        logp = f"{desc.get('LogP', np.nan):.2f}" if desc else "—"
        tpsa = f"{desc.get('TPSA', np.nan):.1f}" if desc else "—"
        metrics_html = f"""
        <div style="display:flex;gap:12px;margin-bottom:10px">
        <div style="flex:0 0 140px;padding:10px;border-radius:10px;background:#fff;border:1px solid #eef2ff">
            <div style="font-size:12px;color:#64748b">MolWt</div>
            <div style="font-weight:700;font-size:16px">{molwt}</div>
        </div>
        <div style="flex:0 0 140px;padding:10px;border-radius:10px;background:#fff;border:1px solid #eef2ff">
            <div style="font-size:12px;color:#64748b">LogP</div>
            <div style="font-weight:700;font-size:16px">{logp}</div>
        </div>
        <div style="flex:0 0 140px;padding:10px;border-radius:10px;background:#fff;border:1px solid #eef2ff">
            <div style="font-size:12px;color:#64748b">TPSA</div>
            <div style="font-weight:700;font-size:16px">{tpsa}</div>
        </div>
        </div>
        """
        suggestions_html = "<ul style='margin-top:6px'>"
        for s in heur[:8]:
            suggestions_html += f"<li style='margin-bottom:6px'>{s}</li>"
        suggestions_html += "</ul>"
        top_paths = sorted(mt.items(), key=lambda kv: kv[1], reverse=True)[:8]
        rows_html = ""
        if top_paths:
            for t, v in top_paths:
                color = "#b91c1c" if v >= float(st.session_state.get("mt_thr", 0.5)) else "#1f77b4"
                rows_html += f"<div style='display:flex;justify-content:space-between;padding:6px 8px;border-bottom:1px solid #f1f5f9'><div style='font-weight:600'>{t}</div><div style='color:{color};font-weight:700'>{v:.3f}</div></div>"
        else:
            rows_html = "<div style='padding:8px;color:#64748b'>No pathways</div>"
        card_full_html = f"""
        <div style="border-radius:12px;padding:18px;background:#ffffff;border:1px solid #e6eef8;box-shadow:0 4px 12px rgba(2,6,23,0.04)">
        {header_html}
        {metrics_html}
        <div style="display:flex;gap:16px">
            <div style="flex:1 1 55%;min-width:260px">
            <div style="font-weight:700;color:#0f172a;margin-bottom:6px">Recommendations</div>
            <div style="color:#0f172a">{suggestions_html}</div>
            </div>
            <div style="flex:1 1 45%;min-width:220px">
            <div style="font-weight:700;color:#0f172a;margin-bottom:6px">Top pathways (risk)</div>
            <div style='border-radius:8px;border:1px solid #eef2ff;overflow:hidden'>{rows_html}</div>
            </div>
        </div>
        <div style='margin-top:12px;color:#475569;font-size:13px'>
            <div style="font-weight:700;margin-bottom:6px">Explainability (top drivers)</div>
            <div>• {' • '.join(st.session_state.get("top_shap_features", ["MolWt", "LogP", "TPSA"]))}</div>
        </div>
        </div>
        """
        # Compose a plain-text compact summary for clipboard copying/export
        summary_lines = [
            f"SMILES: {st.session_state.get('smiles','')}",
            f"Toxicity probability: {bin_prob:.3f}",
            f"Developability: {dev_score:.3f}" if dev_score is not None else "Developability: N/A",
            "Top pathways:"
        ]
        for t, v in top_paths:
            summary_lines.append(f" - {t}: {v:.3f}")
        summary_lines.append("Recommendations:")
        for s in heur[:6]:
            summary_lines.append(f" - {s}")
        summary_text = "\n".join(summary_lines)

        # Copy button under the card
        return card_full_html, summary_text
        return card_full_html
    
    # ---------- Enhanced Knowledge Base Integration ----------
    # Import enhanced KB system with improved content extraction
    try:
        from enhanced_kb_retrieval import EnhancedKBRetrieval
        KB_AVAILABLE = True
    except ImportError:
        KB_AVAILABLE = False
        # Note: Warning will be shown in UI if needed

    # Initialize enhanced KB index globally
    if KB_AVAILABLE:
        @st.cache_resource
        def get_kb_index():
            """Initialize and cache the enhanced KB index for fast repeated access"""
            try:
                enhanced_kb = EnhancedKBRetrieval(kb_dir="kb")
                success = enhanced_kb.build()
                if success:
                    return enhanced_kb
                else:
                    return None
            except Exception as e:
                st.error(f"Failed to initialize enhanced knowledge base: {e}")
                return None
    
        # Load enhanced KB once
        kb_index = get_kb_index() if KB_AVAILABLE else None
        ENABLE_KB = kb_index is not None
    else:
        kb_index = None
        ENABLE_KB = False

    def display_knowledge_base_results(kb_results, title="🧠 Knowledge Base Insights"):
        """Display only the single best knowledge base result with highest relevance"""
        if not kb_results:
            st.info("💡 No relevant references found in the knowledge base for this query.")
            return
        
        st.markdown(f"### {title}")
        
        # Show only the top result with highest score
        result = kb_results[0]
        source = result.get("source", "unknown")
        text = result.get("text", "")
        score = result.get("score", 0.0)
        result_type = result.get("type", "original")
        
        # Clean source name for display
        display_source = source.replace(".md", "").replace("_", " ").title()
        
        # Create confidence indicator
        if score > 0.3:
            confidence_emoji = "🎯"
            confidence_text = "High relevance"
            confidence_color = "#059669"
        elif score > 0.15:
            confidence_emoji = "📍"
            confidence_text = "Medium relevance"
            confidence_color = "#d97706"
        else:
            confidence_emoji = "📋"
            confidence_text = "Low relevance"
            confidence_color = "#6b7280"
        
        # Always expand the single best result
        with st.expander(f"{confidence_emoji} **{display_source}** - {confidence_text} (score: {score:.3f})", expanded=True):
            st.markdown(f"**Source file:** `{source}`")
            st.markdown(f"**Relevance score:** {score:.3f}")
            if result_type == "curated":
                st.markdown("**Type:** Curated answer")
            elif result_type == "extracted":
                st.markdown("**Type:** Extracted content")
            st.markdown("**Content:**")
            st.markdown(f"> {text}")
            
            # Show details if available (for curated answers)
            if result.get("details"):
                st.markdown("**Additional details:**")
                st.markdown(f"> {result['details']}")

    def _format_kb_results_html(kb_results):
        """Format enhanced knowledge base results with beautiful presentation"""
        if not kb_results:
            return '<div style="color:#94a3b8;font-style:italic">No relevant references found.</div>'
        
        # Only show the best result
        result = kb_results[0]
        source = result.get("source", "unknown")
        text = result.get("text", "")
        score = result.get("score", 0.0)
        result_type = result.get("type", "original")
        details = result.get("details", "")
        
        # Clean source name for display
        display_source = source.replace(".md", "").replace("_", " ").title()
        
        # Different handling based on result type
        if result_type == "curated":
            # Curated answers are complete and high-quality
            content = text
            if details:
                content += f"\n\n💡 {details}"
            confidence_color = "#059669"
            confidence_text = "Curated"
            icon = "🎯"
        else:
            # Extracted or original content
            content = text
            confidence_color = "#059669" if score > 0.3 else "#d97706" if score > 0.15 else "#dc2626"
            confidence_text = "High" if score > 0.3 else "Medium" if score > 0.15 else "Low"
            icon = "📄" if result_type == "extracted" else "📝"
        
        # Intelligent text presentation
        if len(content) > 300:
            # Try to cut at sentence boundary
            truncated = content[:300]
            last_period = truncated.rfind('.')
            if last_period > 200:
                content = truncated[:last_period + 1]
            else:
                content = truncated + "..."
        
        # Format content with better markdown handling
        formatted_content = content.replace("**", "<strong>").replace("**", "</strong>")
        formatted_content = formatted_content.replace("\n", "<br>")
        
        html_result = f"""
        <div style="margin-bottom:12px;padding:14px;background:#f8fafc;border-left:4px solid {confidence_color};border-radius:0 8px 8px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-weight:600;color:#1e293b;font-size:13px">{icon} {display_source}</span>
                <span style="font-size:10px;color:{confidence_color};background:{confidence_color}1a;padding:3px 8px;border-radius:10px;font-weight:500">
                    {confidence_text}
                </span>
            </div>
            <div style="color:#475569;font-size:13px;line-height:1.5">{formatted_content}</div>
        </div>
        """
        
        return html_result

    def query_knowledge_base(query: str, top_k: int = 1):
        """Enhanced query that handles multi-concept queries and filters out SMILES strings for better KB search"""
        if not ENABLE_KB or kb_index is None:
            return []
        
        import re
        
        try:
            # Enhanced query preprocessing to improve search relevance
            processed_query = query
            
            # Remove SMILES strings from the query for KB search
            # SMILES typically don't help with conceptual knowledge base searches
            def remove_smiles_from_query(text):
                words = text.split()
                filtered_words = []
                for word in words:
                    # Remove common SMILES-like patterns that aren't useful for KB search
                    clean_word = word.rstrip('.,!?;:')
                    if looks_like_smiles(clean_word):
                        continue  # Skip SMILES strings
                    filtered_words.append(word)
                return ' '.join(filtered_words)
            
            # Clean the query
            processed_query = remove_smiles_from_query(query)
            
            # Also remove common question words that don't help with search
            processed_query = re.sub(r'\b(provide for|show for|give for|for)\b', '', processed_query, flags=re.IGNORECASE)
            processed_query = processed_query.strip()
            
            # First, try the processed query
            results = kb_index.query(processed_query, topk=top_k)
            
            # If we have a multi-concept query (contains 'and', '&', ','), also search individual terms
            multi_indicators = [' and ', ' & ', ', ', ' or ']
            is_multi_concept = any(indicator in processed_query.lower() for indicator in multi_indicators)
            
            if is_multi_concept:
                # Split the query into individual concepts
                # Split by common separators and clean up
                concepts = re.split(r'\s+(?:and|&|,|or)\s+', processed_query.lower())
                concepts = [concept.strip().rstrip('?').strip() for concept in concepts if concept.strip()]
                
                # Search for each concept individually
                for concept in concepts:
                    if len(concept) > 2:  # Only search meaningful terms
                        concept_results = kb_index.query(concept, topk=2)
                        # Add concept results if they're not already included
                        for cr in concept_results:
                            # Check if this result is already in our results (avoid duplicates)
                            if not any(r.get("text", "")[:50] == cr.get("text", "")[:50] for r in results):
                                results.append(cr)
            
            # Format results for compatibility and sort by score
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("score", 0.0),
                    "text": result.get("text", ""),
                    "source": result.get("source", "unknown"),
                    "type": result.get("type", "original")
                })
            
            # Sort by score (highest first) and limit to reasonable number
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results[:min(len(formatted_results), 5)]  # Max 5 results
            
        except Exception as e:
            st.error(f"KB query failed: {e}")
            return []

    # ---------- short format helpers ----------
    def shap_short_phrases(desc_table=None, topn=3):
        if desc_table is None:
            return []
        try:
            df = desc_table.copy()
            df["abs"] = df["shap"].abs()
            df = df.sort_values("abs", ascending=False).head(topn)
            return [f"{r['feature']}({r['shap']:.3f})" for _, r in df.iterrows()]
        except Exception:
            return []

    def compose_answer(user_q, ctx):
        # HEADLINE
        prob = ctx.get("binary_proba", 0.0)
        lab = "TOXIC" if prob >= ctx.get("bin_thr", 0.5) else "NON-TOXIC"
        dev = ctx.get("developability", None)
        devtxt = f"**Developability**: {dev:.2f}." if dev is not None else ""
        header = f"**Quick summary — {lab}** (tox prob={prob:.2f})  \n{devtxt}"

        # WHY / DRIVERS
        parts = []
        mt = ctx.get("mt_probs", {})
        if mt:
            top = sorted(mt.items(), key=lambda kv: kv[1], reverse=True)[:3]
            parts.append("**Top risk pathways:** " + ", ".join([f"{k} ({v:.2f})" for k, v in top]))
        if ctx.get("shap_summary"):
            parts.append("**Model drivers (SHAP):** " + ", ".join(ctx["shap_summary"]))
        if ctx.get("desc"):
            d = ctx["desc"]
            parts.append(f"**Properties:** MW={d.get('MolWt',np.nan):.0f}, LogP={d.get('LogP',np.nan):.2f}, TPSA={d.get('TPSA',np.nan):.1f}.")
        why = "  \n".join(parts) if parts else "No major drivers identified."

        # SUGGESTIONS
        heur = ctx.get("heuristic_recs", [])
        if not heur:
            heur = ["Consider reducing LogP, adding polar groups to improve solubility, replace reactive nitro/aniline groups."]
        suggestions = "\n".join([f"- {h}" for h in heur])

        # KB snippets with enhanced formatting
        kb_snips = ctx.get("kb_snips", [])
        kb_text = ""
        if kb_snips:
            kb_formatted = []
            for snip in kb_snips[:2]:  # Limit to top 2 for text response
                source = snip.get('source', 'unknown').replace('.md', '').replace('_', ' ').title()
                text = snip.get('text', '')[:180] + "..." if len(snip.get('text', '')) > 180 else snip.get('text', '')
                score = snip.get('score', 0.0)
                confidence = "🎯 High" if score > 0.3 else "📍 Medium" if score > 0.15 else "📋 Low"
                kb_formatted.append(f"**{source}** ({confidence} relevance): {text}")
            
            kb_text = "\n\n**🧠 Knowledge Base References:**\n" + "\n".join([f"- {kf}" for kf in kb_formatted])

        # Choose style by intent
        q = (user_q or "").lower()
        if any(tok in q for tok in ("summary","overview","short","brief")):
            body = f"{header}\n\n{why}\n\n**Next steps**\n{suggestions}{kb_text}"
        elif any(tok in q for tok in ("improve","modify","analog","suggest")):
            body = f"{header}\n\n**Actionable modifications**\n{suggestions}\n\n{why}{kb_text}"
        elif any(tok in q for tok in ("why","because","explain")):
            body = f"{why}\n\n{kb_text}"
        else:
            body = f"{header}\n\n{why}\n\n**Suggested changes**\n{suggestions}\n\n{kb_text}"

        return textwrap.dedent(body)

    # ---------- UI layout ----------
    st.markdown('<div style="margin-bottom: -15px;"><h2 style="margin-bottom: 5px;">🧞‍♂️ MolGenie</h2></div>', unsafe_allow_html=True)
    # left, right = st.columns([3, 1], gap="large")
    st.set_page_config(page_title="MolGenie", layout="wide")
    left, right = st.columns([4, 1], gap="large")

    with right:
        st.markdown("**Copilot Controls**")
        include_shap = st.checkbox("Include SHAP in context", value=True, key="copilot_use_shap")
        if st.button("Clear chat", key="copilot_clear"):
            st.session_state["chat_history"] = []
        
        # Knowledge Base Status Indicator
        st.markdown("---")
        st.markdown("**Knowledge Base**")
        if ENABLE_KB:
            st.success("🧠 KB Active - Ready for queries")
        else:
            st.warning("🚫 KB Unavailable")
            st.caption("Install dependencies or check kb/ folder")
        
        st.markdown("---")
        st.markdown("**Quick actions**")
        if st.button("Re-run analysis (predict)"):
            # safe auto-run if predict_all_models exists
            smiles = st.session_state.get("smiles", "")
            if smiles and "predict_all_models" in globals():
                try:
                    preds = predict_all_models(smiles)
                    # NEW: store copilot context only; do NOT set analyzed=True
                    st.session_state["smiles"] = preds["smiles"]
                    st.session_state["mol"] = preds["mol"]
                    st.session_state["binary_proba"] = preds["binary"]["prob"]
                    st.session_state["binary_label"] = preds["binary"]["label"]
                    st.session_state["mt_probs"] = preds["multitask"]
                    st.session_state["developability_score"] = preds["developability"]
                    st.session_state["computed_desc"] = preds["descriptors"]
                    st.session_state["x_mt"] = preds.get("x_feat", None)

                    # marker that copilot has run & context exists (UI-only)
                    st.session_state["copilot_context"] = True
                    st.success("Re-analysis completed.")
                except Exception as e:
                    st.error(f"Re-analysis failed: {e}")
            else:
                st.info("No SMILES in session or predict_all_models() not available.")

    # Chat input area
    # ---------- Left column: single authoritative input (replace your old `with left:`) ----------
    # This block intentionally uses the top-page SMILES as the single source of truth.
    with left:
        #st.markdown("### Copilot — quick interactive analysis (uses the top SMILES input)")
        # show which SMILES is currently authoritative (raw user string)
        current_display = st.session_state.get("smiles_input", "")
        if current_display:
            st.caption(f"Currently selected SMILES (top input): `{current_display}`")

        # A single short text area for optional Copilot question (NOT for the canonical SMILES)
        # If user pastes a SMILES here, we'll canonicalize & copy it into the top input later in the Ask handler.
        copilot_prompt = st.text_area(
            "Ask Copilot (or paste SMILES here) — e.g. 'Summarize risks' or 'CC(=O)O...'",
            key="copilot_input",
            value="",
            height=80
        )

        # Render Ask + Re-run buttons compactly (use unique keys)
        b1, b2, b3 = st.columns([0.18, 0.18, 0.3])
        
        # Simple CSS-only styling for Ask Copilot button - no JavaScript interference
        st.markdown("""
        <style>
        /* Ask Copilot Button - CSS Only */
        div[data-testid="stButton"] > button[key="copilot_ask_btn"],
        div[data-testid="column"] > div[data-testid="stButton"] > button[key="copilot_ask_btn"],
        button[key="copilot_ask_btn"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stButton"] > button[key="copilot_ask_btn"]:hover,
        div[data-testid="column"] > div[data-testid="stButton"] > button[key="copilot_ask_btn"]:hover,
        button[key="copilot_ask_btn"]:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with b1:
            ask = st.button("Ask Copilot", key="copilot_ask_btn")
        # with b2:
        #     rerun = st.button("Re-run analysis", key="copilot_rerun_btn")
        with b3:
            pass  # placeholder for potential future button
            
        # NOTE: The actual Ask handler below should use the canonicalization logic we discussed
        # and update st.session_state['smiles_input'] (raw) and st.session_state['smiles'] (canonical).
        # (Do not set st.session_state['analyzed']=True here; user must click Show.)

        # Load KB
        # KB is now automatically loaded when the app starts via @st.cache_resource

        # Initialize history
        st.session_state.setdefault("chat_history", [])
        st.session_state.setdefault("copilot_context", False)
        st.session_state.setdefault("analyzed", False)
        st.session_state.setdefault("text_only_query", False)

        # Handle Ask
        # -------- Enhanced Three-Way Ask handler --------
        if ask:
            user_q = (copilot_prompt or "").strip()

            # Helper function to detect SMILES
            def looks_like_smiles(s):
                """Detect if a string looks like a SMILES molecule"""
                if not s or len(s) <= 2:
                    return False
                
                # Exclude common chemistry terms and English words that aren't SMILES (case-insensitive)
                chemistry_terms = {
                    'smiles', 'adme', 'admet', 'tpsa', 'logp', 'logd', 'mw', 'molwt',
                    'psa', 'hba', 'hbd', 'hacc', 'hdon', 'toxicity', 'solubility', 
                    'permeability', 'bioavailability', 'bbbp', 'notation', 'analysis', 
                    'prediction', 'molecular', 'compound', 'drug', 'molecule', 'chemical', 
                    'structure', 'formula', 'properties', 'toxic', 'analyze', 'what', 
                    'how', 'why', 'when', 'where', 'which', 'is', 'are', 'does', 'do', 
                    'can', 'will', 'would', 'should', 'and', 'or', 'but', 'for', 'of', 
                    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'with', 'by',
                    'explain', 'describe', 'tell', 'mean', 'means', 'definition',
                    'lipophilicity', 'polarity', 'weight', 'mass', 'area', 'surface',
                    'show', 'it', 'standard', 'gold', 'considered', 'tox21', 'predictive'
                }
                
                # Check if it's a known chemistry term (case-insensitive)
                if s.lower() in chemistry_terms:
                    return False
                
                # Additional check: if it's a common English word pattern, it's not SMILES
                s_lower = s.lower()
                if any(pattern in s_lower for pattern in ['qu', 'th', 'tion', 'ness', 'ing', 'ed', 'ly', 'er', 'est', 'ow']):
                    return False
                
                # Check if it looks like a SMILES string
                # Must start with typical SMILES atoms and have SMILES-like characteristics
                if not (s[0] in "CNOPSFClBrI[cnops" and " " not in s and "?" not in s):
                    return False
                
                # Additional validation: check for SMILES patterns
                # Real SMILES often have parentheses, brackets, or numbers
                smiles_patterns = ['(', ')', '[', ']', '=', '#', '1', '2', '3', '4', '5', '6']
                has_smiles_patterns = any(pattern in s for pattern in smiles_patterns)
                
                # For simple cases like "CCO", allow even without special patterns
                # but be more strict for longer strings
                if len(s) <= 4:
                    return True  # Short strings like CCO, CO, etc.
                else:
                    return has_smiles_patterns  # Longer strings need SMILES patterns
            
            # Helper function to extract SMILES from text
            def extract_smiles_from_text(text):
                """Extract SMILES from text like 'what is toxicity of CCO'"""
                words = text.split()
                potential_smiles = []
                
                for word in words:
                    # Remove punctuation from word for SMILES detection
                    clean_word = word.rstrip('.,!?;:')
                    if looks_like_smiles(clean_word):
                        potential_smiles.append(clean_word)
                
                # Return the longest valid SMILES (most likely to be the actual molecule)
                if potential_smiles:
                    # Sort by length and return the longest one (more likely to be the real SMILES)
                    return max(potential_smiles, key=len)
                return None
            
            # Determine the scenario based on input analysis
            pure_smiles = looks_like_smiles(user_q)
            embedded_smiles = extract_smiles_from_text(user_q) if not pure_smiles else None
            has_text_question = bool(user_q) and not pure_smiles
            
            # Get SMILES from top input if available
            top_smiles_raw = st.session_state.get("smiles_input", "")
            top_smiles = (top_smiles_raw or "").strip()
            
            # SCENARIO 1: Pure SMILES input - show summary + detailed analysis button
            if pure_smiles:
                raw_input = user_q
                
                # Validate and canonicalize SMILES
                mol_tmp = Chem.MolFromSmiles(raw_input)
                if mol_tmp is None:
                    st.error("❌ Invalid SMILES string. Please check and try again.")
                    return
                
                canonical_smiles = Chem.MolToSmiles(mol_tmp)
                
                # Show scenario label
                st.info("🧬 **Scenario 1**: Pure SMILES input detected - Molecular analysis")
                
                # Update session state for detailed analysis
                st.session_state["smiles_input"] = raw_input
                st.session_state["smiles"] = canonical_smiles
                st.session_state["text_only_query"] = False
                st.session_state["copilot_context"] = True
                st.session_state["pure_smiles_mode"] = True  # Flag for Scenario 1
                
                # Run molecular analysis for summary
                try:
                    preds = predict_all_models(canonical_smiles)
                    
                    # Store predictions in session state
                    st.session_state["mol"] = preds["mol"]
                    st.session_state["binary_proba"] = preds["binary"]["prob"]
                    st.session_state["binary_label"] = preds["binary"]["label"]
                    st.session_state["mt_probs"] = preds["multitask"]
                    st.session_state["developability_score"] = preds["developability"]
                    st.session_state["computed_desc"] = preds["descriptors"]
                    st.session_state["x_mt"] = preds.get("x_feat", None)
                    st.session_state["top_shap_features"] = preds.get("top_shap_features", ["MolWt", "LogP", "TPSA"])
                    
                    # Toxicity summary
                    binary_prob = preds["binary"]["prob"]
                    is_toxic = binary_prob > 0.5
                    
                except Exception as e:
                    st.error(f"Error generating molecular analysis: {e}")
            
            # SCENARIO 2: Pure text question - KB search only
            elif has_text_question and not embedded_smiles:
                st.info("💭 **Scenario 2**: Text question detected - Searching knowledge base")
                
                try:
                    with st.spinner(f"🔍 Searching knowledge base for: '{user_q}'..."):
                        kb_results = query_knowledge_base(user_q, top_k=1)  # Only get the single best result
                    
                    st.session_state["text_only_query"] = True
                    st.session_state["copilot_context"] = False
                    st.session_state["pure_smiles_mode"] = False  # Not pure SMILES
                    
                    if kb_results:
                        st.success("✅ **Found relevant information in knowledge base**")
                        display_knowledge_base_results(kb_results, f"📖 Answer: '{user_q}'")
                        st.info("💡 **Tip**: To analyze a specific molecule, include a SMILES string in your question or use the top input.")
                    else:
                        st.warning(f"🔍 **No relevant information found for**: '{user_q}'")
                        st.markdown("""
                        **Try these suggestions:**
                        - Use terms like 'toxicity', 'LogP', 'TPSA', 'solubility', 'BBBP'
                        - Ask about molecular properties or drug discovery concepts
                        - Include a SMILES string for molecular-specific analysis
                        """)
                        
                except Exception as e:
                    st.error(f"Knowledge base search failed: {e}")
                
                return
            
            # SCENARIO 3: Text question with embedded SMILES - KB answer + detailed analysis option
            elif has_text_question and embedded_smiles:
                st.info("🔬💭 **Scenario 3**: Text question with SMILES detected - Dual analysis")
                
                # Extract and validate SMILES
                mol_tmp = Chem.MolFromSmiles(embedded_smiles)
                if mol_tmp is None:
                    st.error(f"❌ Invalid SMILES '{embedded_smiles}' found in your question. Please check and try again.")
                    return
                
                canonical_smiles = Chem.MolToSmiles(mol_tmp)
                
                # Update session state for detailed analysis capability
                st.session_state["smiles_input"] = embedded_smiles
                st.session_state["smiles"] = canonical_smiles
                st.session_state["text_only_query"] = False
                st.session_state["copilot_context"] = True
                st.session_state["pure_smiles_mode"] = False  # Not pure SMILES (has text question)
                
                # Show extracted SMILES
                st.success(f"🧬 **Extracted SMILES**: `{canonical_smiles}`")
                
                # Answer the text question from KB
                try:
                    # Clean the query for better KB search by focusing on the conceptual question
                    # Remove the extracted SMILES from the question for better KB search
                    kb_query = user_q
                    if embedded_smiles in user_q:
                        kb_query = user_q.replace(embedded_smiles, '').strip()
                    
                    # Clean up any leftover patterns
                    import re
                    kb_query = re.sub(r'\s+', ' ', kb_query)  # Remove extra spaces
                    kb_query = re.sub(r'\b(provide for|show for|give for|for)\s*$', '', kb_query, flags=re.IGNORECASE).strip()
                    
                    with st.spinner(f"🔍 Searching knowledge base for: '{kb_query}'..."):
                        kb_results = query_knowledge_base(kb_query, top_k=3)  # Get more results for comprehensive answers
                    
                    if kb_results:
                        st.markdown("### 💭 **Answer to your question:**")
                        display_knowledge_base_results(kb_results, f"📖 KB Response")
                    else:
                        st.warning("🔍 No specific answer found in knowledge base for your question.")
                        
                except Exception as e:
                    st.error(f"Knowledge base search failed: {e}")
                
                # Run predictions for the extracted SMILES to enable detailed analysis
                try:
                    preds = predict_all_models(canonical_smiles)
                    
                    # Store predictions in session state for detailed analysis
                    st.session_state["mol"] = preds["mol"]
                    st.session_state["binary_proba"] = preds["binary"]["prob"]
                    st.session_state["binary_label"] = preds["binary"]["label"]
                    st.session_state["mt_probs"] = preds["multitask"]
                    st.session_state["developability_score"] = preds["developability"]
                    st.session_state["computed_desc"] = preds["descriptors"]
                    st.session_state["x_mt"] = preds.get("x_feat", None)
                    st.session_state["top_shap_features"] = preds.get("top_shap_features", ["MolWt", "LogP", "TPSA"])
                    
                except Exception as e:
                    st.error(f"Error analyzing extracted SMILES: {e}")
            
            else:
                # No valid input provided
                st.warning("❓ Please provide either:")
                st.markdown("""
                - **A SMILES string** (e.g., `CCO`) for molecular analysis
                - **A text question** (e.g., "What is TPSA?") for knowledge base search  
                - **A question with SMILES** (e.g., "Is CCO toxic?") for combined analysis
                """)
                return
                # ------------------ end Ask block ------------------

                # Always set copilot_context True after successful analysis
                #st.session_state["copilot_context"] = True

                # Build context
                ctx = {
                    "binary_proba": float(st.session_state.get("binary_proba", 0.0)),
                    "bin_thr": float(st.session_state.get("bin_thr", 0.5)),
                    "mt_probs": st.session_state.get("mt_probs", {}) or {},
                    "developability": st.session_state.get("developability_score", None),
                    "desc": st.session_state.get("computed_desc", {}) or {},
                    "shap_summary": shap_short_phrases(st.session_state.get("desc_table", None), topn=4) if include_shap else [],
                    "heuristic_recs": heuristic_recs(st.session_state.get("computed_desc", {}), st.session_state.get("mt_probs", {}), thr=st.session_state.get("bin_thr",0.5)) if "heuristic_recs" in globals() else []
                }

                # KB retrieval using optimized system
                kb_query = (user_q or "") + " " + " ".join(ctx["shap_summary"])
                kb_res = query_knowledge_base(kb_query, top_k=1)
                ctx["kb_snips"] = kb_res
                
                # Store KB results for display in UI
                st.session_state._last_kb_results = kb_res
                st.session_state._last_kb_query = kb_query


                # Fix variable scope for summary text
                bin_prob = float(ctx.get("binary_proba", 0.0))
                dev_score = ctx.get("developability", None)
                mt = ctx.get("mt_probs", {}) or {}
                heur = ctx.get("heuristic_recs", []) or []
                top_paths = sorted(mt.items(), key=lambda kv: kv[1], reverse=True)[:8]

                # Compose a plain-text compact summary for clipboard copying/export
                summary_lines = [
                    f"SMILES: {st.session_state.get('smiles','')}",
                    f"Toxicity probability: {bin_prob:.3f}",
                    f"Developability: {dev_score:.3f}" if dev_score is not None else "Developability: N/A",
                    "Top pathways:"
                ]
                for t, v in top_paths:
                    summary_lines.append(f" - {t}: {v:.3f}")
                summary_lines.append("Recommendations:")
                for s in heur[:6]:
                    summary_lines.append(f" - {s}")
                summary_text = "\n".join(summary_lines)

                # --------- END: polished result rendering ----------


                # Compose answer and append to history
                # answer = compose_answer(user_q or "Summarize this molecule", ctx)
                # now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                #st.session_state["chat_history"].append({"who":"user","text":user_q or smiles_guess, "time": now})
                #st.session_state["chat_history"].append({"who":"copilot","text":answer, "time": now, "kb": kb_res})

                # Optionally store a short transcript file locally for audit
                # try:
                #     os.makedirs("chat_logs", exist_ok=True)
                #     logpath = os.path.join("chat_logs", f"chat_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
                #     with open(logpath, "w", encoding="utf-8") as fh:
                #         json.dump(st.session_state["chat_history"], fh, indent=2)
                # except Exception:
                #     pass

        # Robust chat-history renderer: supports both old tuple entries and new dict entries.
        # history = st.session_state.get("chat_history", [])
                # Fix variable scope for summary text
                bin_prob = float(ctx.get("binary_proba", 0.0))
                dev_score = ctx.get("developability", None)
                mt = ctx.get("mt_probs", {}) or {}
                heur = ctx.get("heuristic_recs", []) or []
                top_paths = sorted(mt.items(), key=lambda kv: kv[1], reverse=True)[:8]

        # # show most recent first
        # for entry in history[::-1]:
        #     # Normalize entry into a dict for uniform handling
        #     if isinstance(entry, dict):
        #         e = entry
        #     elif isinstance(entry, (list, tuple)):
        #         # tuple formats we expect:
        #         # ("You", text) or ("You", text, time) or ("Medic Copilot", text, {"kb": ...})
        #         who = entry[0] if len(entry) > 0 else "user"
        #         text = entry[1] if len(entry) > 1 else ""
        #         # try to pull time/meta if present
        #         meta = None
        #         tstamp = ""
        #         if len(entry) > 2:
        #             if isinstance(entry[2], str):
        #                 tstamp = entry[2]
        #             elif isinstance(entry[2], dict):
        #                 meta = entry[2]
        #                 tstamp = entry[2].get("time", "")
        #         e = {"who": who if isinstance(who, str) else str(who),
        #             "text": text if text is not None else "",
        #             "time": tstamp,
        #             "kb": meta.get("kb") if meta and isinstance(meta, dict) else (meta if isinstance(meta, list) else [])}
        #     else:
        #         # Unknown entry type — stringify it
        #         e = {"who": "system", "text": str(entry), "time": ""}

        #     who = e.get("who", "user")
        #     text = e.get("text", "")
        #     tstamp = e.get("time", "")
        #     kbmeta = e.get("kb", []) or []

        #     # Render nicely (use st.chat_message if available)
        #     if hasattr(st, "chat_message"):
        #         role = "user" if str(who).lower().startswith("you") or str(who).lower()=="user" else "assistant"
        #         with st.chat_message(role):
        #             if tstamp:
        #                 st.markdown(f"**{who.title()}** • {tstamp}")
        #             else:
        #                 st.markdown(f"**{who.title()}**")
        #             st.markdown(text)
        #             if kbmeta:
        #                 st.caption("Sources: " + ", ".join([m.get("source", str(m)) if isinstance(m, dict) else str(m) for m in kbmeta]))
        #             # small action buttons (non-blocking)
        #             cols = st.columns([0.15, 2.85])
        #             try:
        #                 with cols[0]:
        #                     # safe unique key based on hash
        #                     render_copy_button(text, key=str(abs(hash(text))))
        #                 # leave the larger column empty to preserve layout spacing (or show tiny metadata)
        #                 with cols[1]:
        #                     st.caption("")  # no export button; keep spacing clean
        #             except Exception:
        #                 pass

        #     else:
        #         # fallback bubble layout if st.chat_message not available
        #         if str(who).lower().startswith("you") or str(who).lower()=="user":
        #             c1, c2 = st.columns([0.2, 3])
        #             with c2:
        #                 st.markdown(f"**You** {('• ' + tstamp) if tstamp else ''}")
        #                 st.markdown(f"> {text}")
        #         else:
        #             c1, c2 = st.columns([3, 0.2])
        #             with c1:
        #                 st.markdown(f"**{who}** {('• ' + tstamp) if tstamp else ''}")
        #                 st.markdown(text)
        #                 if kbmeta:
        #                     st.caption("Sources: " + ", ".join([m.get("source", str(m)) if isinstance(m, dict) else str(m) for m in kbmeta]))

        if st.session_state.get("copilot_context", False) and not st.session_state.get("text_only_query", False):
            card_full_html, summary_text = build_copilot_card()
            c1, c2 = st.columns([0.02, 0.98])
            with c1:
                st.markdown("")  # tiny gutter
            with c2:
                st.markdown(card_full_html, unsafe_allow_html=True)
                render_copy_button(summary_text, key=f"copy_summary_{abs(hash(summary_text))}")
            
            # Display Knowledge Base Results with beautiful formatting
            if ENABLE_KB:
                # Get KB results only for scenarios 2 and 3, not for pure SMILES input (scenario 1)
                current_kb_results = []
                try:
                    # Only show KB insights if we have KB results from text queries
                    # Don't generate KB for pure SMILES input (Scenario 1)
                    if hasattr(st.session_state, '_last_kb_results') and not st.session_state.get("pure_smiles_mode", False):
                        current_kb_results = st.session_state._last_kb_results
                    # Don't auto-generate KB queries for pure SMILES scenarios
                        
                except Exception as e:
                    st.caption(f"KB query issue: {e}")
                    current_kb_results = []
                
                if current_kb_results:
                    st.markdown("---")
                    c1_kb, c2_kb = st.columns([0.02, 0.98])
                    with c1_kb:
                        st.markdown("")  # tiny gutter
                    with c2_kb:
                        display_knowledge_base_results(current_kb_results, "🧠 Knowledge Base Insights")
            
            # Show the Show Detailed Analysis button only if not analyzed AND we have molecular data (not text-only query)
            if not st.session_state.get("analyzed", False) and not st.session_state.get("text_only_query", False) and st.session_state.get("copilot_context", False):
                c1, c2 = st.columns([0.18, 0.82])
                with c1:
                    # Style the detailed analysis button with beautiful gradient and JavaScript
                    st.markdown("""
                    <style>
                    /* Enhanced Show Detailed Analysis Button - Modern Gradient Design */
                    div[data-testid="stButton"] > button[key="btn_show_analysis_small"],
                    div[data-testid="column"] > div[data-testid="stButton"] > button[key="btn_show_analysis_small"],
                    button[key="btn_show_analysis_small"] {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                        color: white !important;
                        border: none !important;
                        border-radius: 12px !important;
                        font-weight: 700 !important;
                        padding: 12px 24px !important;
                        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        font-size: 14px !important;
                        letter-spacing: 0.5px !important;
                        text-transform: uppercase !important;
                        position: relative !important;
                        overflow: hidden !important;
                        width: 100% !important;
                    }
                    
                    div[data-testid="stButton"] > button[key="btn_show_analysis_small"]:hover,
                    div[data-testid="column"] > div[data-testid="stButton"] > button[key="btn_show_analysis_small"]:hover,
                    button[key="btn_show_analysis_small"]:hover {
                        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
                        transform: translateY(-3px) scale(1.02) !important;
                        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
                    }
                    
                    div[data-testid="stButton"] > button[key="btn_show_analysis_small"]:active,
                    button[key="btn_show_analysis_small"]:active {
                        transform: translateY(-1px) scale(0.98) !important;
                        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
                    }
                    
                    /* Add ripple effect */
                    div[data-testid="stButton"] > button[key="btn_show_analysis_small"]::before,
                    button[key="btn_show_analysis_small"]::before {
                        content: '' !important;
                        position: absolute !important;
                        top: 0 !important;
                        left: -100% !important;
                        width: 100% !important;
                        height: 100% !important;
                        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
                        transition: left 0.5s ease !important;
                    }
                    
                    div[data-testid="stButton"] > button[key="btn_show_analysis_small"]:hover::before,
                    button[key="btn_show_analysis_small"]:hover::before {
                        left: 100% !important;
                    }
                    </style>
                    
                    <script>
                    setTimeout(function() {
                        // Find Show Detailed Analysis button and apply enhanced styling directly
                        const buttons = document.querySelectorAll('button');
                        for (const btn of buttons) {
                            if (btn.textContent && btn.textContent.includes('Show detailed analysis')) {
                                btn.style.cssText = `
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                                    color: white !important;
                                    border: none !important;
                                    border-radius: 12px !important;
                                    font-weight: 700 !important;
                                    padding: 12px 24px !important;
                                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
                                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                                    font-size: 14px !important;
                                    letter-spacing: 0.5px !important;
                                    text-transform: uppercase !important;
                                    position: relative !important;
                                    overflow: hidden !important;
                                    width: 100% !important;
                                `;
                                
                                btn.addEventListener('mouseenter', function() {
                                    this.style.background = 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)';
                                    this.style.transform = 'translateY(-3px) scale(1.02)';
                                    this.style.boxShadow = '0 12px 35px rgba(102, 126, 234, 0.6)';
                                });
                                
                                btn.addEventListener('mouseleave', function() {
                                    this.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                                    this.style.transform = 'translateY(0px) scale(1)';
                                    this.style.boxShadow = '0 8px 25px rgba(102, 126, 234, 0.4)';
                                });
                                
                                btn.addEventListener('mousedown', function() {
                                    this.style.transform = 'translateY(-1px) scale(0.98)';
                                    this.style.boxShadow = '0 5px 15px rgba(102, 126, 234, 0.4)';
                                });
                                
                                btn.addEventListener('mouseup', function() {
                                    this.style.transform = 'translateY(-3px) scale(1.02)';
                                    this.style.boxShadow = '0 12px 35px rgba(102, 126, 234, 0.6)';
                                });
                                break;
                            }
                        }
                    }, 100);
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Info message above the button
                    st.markdown("💡 **Click 'Show detailed analysis' below for comprehensive toxicity predictions and explainability.**")
                    
                    if st.button("🔎 Show detailed analysis", key="btn_show_analysis_small"):
                        # Add loading feedback
                        with st.spinner("Loading detailed analysis..."):
                            # reveal ADMET + request scroll to DrugSafe AI Platform section
                            st.session_state["analyzed"] = True
                            st.session_state["scroll_to_admet"] = True
                            st.rerun()
                with c2:
                    # Enhanced styling for "Ready for" with highlighting
                    smiles_value = st.session_state.get('smiles_input', '(no input)')
                    if smiles_value and smiles_value != '(no input)':
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                            border-left: 4px solid #667eea;
                            padding: 8px 12px;
                            border-radius: 6px;
                            margin-top: 8px;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        ">
                            <strong style="color: #1976d2;">Ready for:</strong> 
                            <code style="
                                background: rgba(102, 126, 234, 0.1);
                                padding: 2px 6px;
                                border-radius: 4px;
                                color: #667eea;
                                font-weight: 600;
                            ">{smiles_value}</code>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
                            border-left: 4px solid #bdbdbd;
                            padding: 8px 12px;
                            border-radius: 6px;
                            margin-top: 8px;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        ">
                            <strong style="color: #757575;">Ready for:</strong> 
                            <code style="
                                background: rgba(189, 189, 189, 0.1);
                                padding: 2px 6px;
                                border-radius: 4px;
                                color: #9e9e9e;
                                font-style: italic;
                            ">{smiles_value}</code>
                        </div>
                        """, unsafe_allow_html=True)


    # close wrapper
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# UI
# -----------------------------

# Add global CSS for clean, sophisticated layout
st.markdown("""
<style>
/* Expand container to prevent text cutoff and add footer space */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Ensure adequate header space */
.main > div {
    padding-top: 0.5rem !important;
}

/* Remove extra spacing from header and footer */
header[data-testid="stHeader"] {
    height: 0px !important;
}

footer {
    visibility: hidden !important;
    height: 1rem !important;
}

/* Reduce spacing around components */
.stMarkdown {
    margin-bottom: 0.25rem !important;
}

/* Aggressive spacing reduction between text input and copilot */
.stTextInput {
    margin-bottom: -10px !important;
}

.stTextInput + .element-container {
    margin-top: -15px !important;
}

/* Target copilot container specifically */
#copilot_container {
    margin-top: -25px !important;
    padding-top: 0px !important;
}

/* Reduce spacing after text input specifically */
div[data-testid="stTextInput"] {
    margin-bottom: -10px !important;
}

/* Target the parent container of copilot */
.element-container:has(#copilot_container) {
    margin-top: -20px !important;
}

/* Clean button styling with consistent colors */
.stButton > button {
    border-radius: 6px !important;
    border: 1px solid #e1e5e9 !important;
    padding: 0.375rem 0.75rem !important;
    font-weight: 500 !important;
    background-color: #f8f9fa !important;
    color: #495057 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #e9ecef !important;
    border-color: #adb5bd !important;
}

/* Reduce spacing between elements */
.element-container {
    margin-bottom: 0.25rem !important;
}

/* Tighter text input styling */
.stTextInput > div > div > input {
    padding: 0.375rem 0.75rem !important;
}

/* Compact radio button styling */
.stRadio > div {
    gap: 0.5rem !important;
}

/* Reduce slider spacing */
.stSlider {
    margin-bottom: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- Single authoritative top SMILES input (only one on the page) ---
st.session_state.setdefault("smiles_input", "")   # human-entered string (UI display)
top_val = st.session_state.get("smiles_input", "")

# Apply sophisticated spacing based on page state - no jumping space
if st.session_state.get("analyzed", False):
    # Completely eliminate any spacing gaps in detailed analysis
    st.markdown('<div style="margin-top: -45px; margin-bottom: -15px;"></div>', unsafe_allow_html=True)
    
    # Show SMILES input only on detailed analysis page
    top_smiles = st.text_input("Enter SMILES (top input)", value=top_val, key="top_smiles_input")
    # Add scroll anchor for SMILES input with tighter spacing
    st.markdown('<div id="smiles-input-section" style="margin-bottom: -20px;"></div>', unsafe_allow_html=True)
    # keep the canonical UI variable name in session_state for backward compatibility
    st.session_state["smiles_input"] = top_smiles
else:
    # Clean spacing for main page (no SMILES input on copilot page)
    st.markdown('<div style="margin-top: -30px; margin-bottom: -10px;"></div>', unsafe_allow_html=True)
    # Ensure we maintain the current value even when input is not shown
    if "smiles_input" not in st.session_state:
        st.session_state["smiles_input"] = ""    

# show copilot first; if analyzed, show it and the detailed tabs
if not st.session_state.get("analyzed", False):
    medic_copilot_ui()
else:
    # When analyzed=True, show back button with predict and clear buttons in same row with minimal spacing
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])  # Equal width columns for proper alignment
    
    with button_col1:
        # Style for Back to Copilot button with beautiful gradient and reduced top spacing
        st.markdown("""
        <style>
        /* Reduce spacing above button container */
        div[data-testid="column"]:has(button[key="back_to_copilot"]) {
            margin-top: -40px !important;
        }
        div[data-testid="stButton"] > button[key="back_to_copilot"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            height: 2.5rem !important;
            margin-top: -20px !important;
        }
        div[data-testid="stButton"] > button[key="back_to_copilot"]:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("⬅️ Back to Copilot", key="back_to_copilot"):
            # Clear all copilot-related session state to remove data from copilot page
            st.session_state["analyzed"] = False
            st.session_state["scroll_to_admet"] = False
            
            # Clear copilot scenario flags
            st.session_state["copilot_context"] = False
            st.session_state["text_only_query"] = False
            st.session_state["pure_smiles_mode"] = False
            
            # Clear molecular analysis results
            if "mol" in st.session_state:
                del st.session_state["mol"]
            if "binary_proba" in st.session_state:
                del st.session_state["binary_proba"]
            if "binary_label" in st.session_state:
                del st.session_state["binary_label"]
            if "mt_probs" in st.session_state:
                del st.session_state["mt_probs"]
            if "developability_score" in st.session_state:
                del st.session_state["developability_score"]
            if "computed_desc" in st.session_state:
                del st.session_state["computed_desc"]
            if "x_mt" in st.session_state:
                del st.session_state["x_mt"]
            
            # Clear KB results
            if "_last_kb_results" in st.session_state:
                del st.session_state["_last_kb_results"]
            
            # Clear SMILES input from copilot AND top input field
            if "smiles" in st.session_state:
                st.session_state["smiles"] = ""
            if "smiles_input" in st.session_state:
                st.session_state["smiles_input"] = ""  # Clear the top input field
                
            st.rerun()
    
    with button_col2:
        # Apply comprehensive styling for the red predict button
        st.markdown("""
        <style>
        /* More specific targeting for the predict button */
        div[data-testid="stButton"] > button[key="btn_predict_detailed"],
        div[data-testid="column"] button[key="btn_predict_detailed"],
        button[key="btn_predict_detailed"] {
            background-color: #ff4b4b !important;
            background: #ff4b4b !important;
            color: white !important;
            border: 2px solid #ff4b4b !important;
            font-weight: 600 !important;
            width: 100% !important;
            height: 2.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        /* Hover effects with higher specificity */
        div[data-testid="stButton"] > button[key="btn_predict_detailed"]:hover,
        div[data-testid="column"] button[key="btn_predict_detailed"]:hover,
        button[key="btn_predict_detailed"]:hover {
            background-color: #ff3030 !important;
            background: #ff3030 !important;
            border-color: #ff3030 !important;
            box-shadow: 0 4px 8px rgba(255, 75, 75, 0.4) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Active state */
        div[data-testid="stButton"] > button[key="btn_predict_detailed"]:active,
        button[key="btn_predict_detailed"]:active {
            background-color: #cc2020 !important;
            transform: translateY(0px) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create the button
        st.session_state.detailed_predict_clicked = st.button("🔬 Predict", key="btn_predict_detailed")
        
        # Additional JavaScript for dynamic styling (fallback)
        st.markdown("""
        <script>
        function styleRedButton() {
            const selectors = [
                'button[key="btn_predict_detailed"]',
                'div[data-testid="stButton"] button:has-text("🔬 Predict")',
                'button:contains("🔬 Predict")'
            ];
            
            let button = null;
            for (const selector of selectors) {
                try {
                    if (selector.includes(':contains') || selector.includes(':has-text')) {
                        // Handle text-based selection
                        const buttons = document.querySelectorAll('button');
                        for (const btn of buttons) {
                            if (btn.textContent && btn.textContent.includes('🔬 Predict')) {
                                button = btn;
                                break;
                            }
                        }
                    } else {
                        button = document.querySelector(selector);
                    }
                    if (button) break;
                } catch (e) {
                    console.log('Selector failed:', selector);
                }
            }
            
            if (button) {
                // Apply styles directly
                button.style.cssText = `
                    background-color: #ff4b4b !important;
                    color: white !important;
                    border: 2px solid #ff4b4b !important;
                    font-weight: 600 !important;
                    box-shadow: 0 2px 4px rgba(255, 75, 75, 0.3) !important;
                    transition: all 0.3s ease !important;
                `;
                
                // Add hover effects
                button.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = '#ff3030';
                    this.style.borderColor = '#ff3030';
                    this.style.boxShadow = '0 4px 8px rgba(255, 75, 75, 0.4)';
                    this.style.transform = 'translateY(-1px)';
                });
                
                button.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '#ff4b4b';
                    this.style.borderColor = '#ff4b4b';
                    this.style.boxShadow = '0 2px 4px rgba(255, 75, 75, 0.3)';
                    this.style.transform = 'translateY(0px)';
                });
                
                console.log('Red button styling applied successfully');
            } else {
                console.log('Red predict button not found');
            }
        }
        
        // Try multiple times to catch the button
        setTimeout(styleRedButton, 100);
        setTimeout(styleRedButton, 500);
        setTimeout(styleRedButton, 1000);
        
        // Also try when page is fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', styleRedButton);
        } else {
            styleRedButton();
        }
        </script>
        """, unsafe_allow_html=True)
    
    with button_col3:
        # Style for Clear Results button with beautiful gradient
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[key="btn_clear_detailed"] {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            height: 2.5rem !important;
        }
        div[data-testid="stButton"] > button[key="btn_clear_detailed"]:hover {
            background: linear-gradient(135deg, #ee5a52 0%, #ff6b6b 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🗑️ Clear Results", key="btn_clear_detailed"):
            cleanup_session_state()
            reset_for_new_molecule()
            st.rerun()

if st.session_state.get("analyzed", False):
    # ADMET top: title with tight description - DEMO VERSION
    current_display = st.session_state.get("smiles_input", "")
    smiles_caption = f'<div style="margin:0; padding:0; font-size:12px; color:#666; margin-top:8px;">Currently selected SMILES (top input): <code>{current_display}</code></div>' if current_display else ''
    
    st.markdown(f'''
    <div id="admet_container" data-section="drugsafe-ai-platform" style="margin-bottom: 0px;">
        <h1 id="drugsafe_ai_title" style="margin:0; padding:0; margin-bottom: 4px;">🛡️ DrugSafe AI Platform</h1>
        <div style="margin:0; padding:0; font-size:14px; color:#334155; margin-bottom: 8px;">Intelligent molecular safety and developability prediction platform.</div>
        {smiles_caption}
    </div>
    ''', unsafe_allow_html=True)

    # ---------- Session state init ----------

    # ---- init state
    st.session_state.setdefault("active_tab", "Binary")

    for k, v in {
        "smiles": "",
        "binary_proba": None,
        "binary_label": None,
        "mt_probs": None,
        "mol": None,
        "x_mt": None,
        "last_error": None,
        "explain_task": None,
        "explain_img": None,
        "desc_table": None,
        "show_preview":False,
        "detailed_predict_clicked": False,
        "regular_predict_clicked": False
    }.items():
        st.session_state.setdefault(k, v)

    # ---------- Inputs ----------
    col_input, col_info = st.columns([1.2, 1])
    with col_input:
        
        #st.session_state.smiles = st.text_input("Enter SMILES", st.session_state.smiles)
        threshold = st.slider("Binary decision threshold (toxic if prob ≥ threshold)", 0.0, 1.0, 0.50, 0.01, key="bin_thr")
        mt_thresh = st.slider("Multi-task risk threshold", 0.0, 1.0, 0.50, 0.01, key="mt_thr")

    with col_info:
        st.markdown("**Model summary**")
        st.write("- **Binary model:** 2048 FP + 5 descriptors")
        st.write("- **Multi-task models:** GBM per Tox21 assay (FP + 5 desc)")
        st.write("- **Splits:** Murcko scaffold")


    def render_smiles_rdkitjs(smiles: str, width: int = 260, height: int = 200):
        """Optimized molecule rendering with fallback"""
        cid = f"mol_{hash(smiles) % 10000}"
        smiles_js = json.dumps(smiles)

        html = f"""
            <div id="{cid}" style="width:{width}px;height:{height}px;border:1px solid #eee;background:#f9f9f9;">
                <div style="padding:10px;text-align:center;color:#666;font-size:12px;">Loading...</div>
            </div>
            <script>
            (function() {{
                const target = document.getElementById("{cid}");
                if (!target) return;
                
                let timeout = setTimeout(function() {{
                    target.innerHTML = '<div style="padding:10px;text-align:center;color:#666;font-size:12px;">Molecule: {smiles[:20]}...</div>';
                }}, 5000);
                
                async function loadAndRender() {{
                    try {{
                        let RDKit;
                        if (window.RDKit) {{
                            RDKit = window.RDKit;
                        }} else if (window.initRDKitModule) {{
                            RDKit = await initRDKitModule();
                            window.RDKit = RDKit;
                        }} else {{
                            await new Promise((res, rej) => {{
                                const s = document.createElement('script');
                                s.src = "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js";
                                s.onload = res; s.onerror = rej;
                                document.head.appendChild(s);
                            }});
                            RDKit = await initRDKitModule();
                            window.RDKit = RDKit;
                        }}
                        
                        const mol = RDKit.get_mol({smiles_js});
                        if (!mol) throw new Error("Invalid SMILES");
                        
                        const svg = mol.get_svg_with_highlights(JSON.stringify({{
                            width: {width}, height: {height},
                            kekulize: true, bondLineWidth: 1,
                            backgroundColour: [1,1,1,0]
                        }}));
                        
                        clearTimeout(timeout);
                        target.innerHTML = svg;
                        mol.delete();
                        
                    }} catch (e) {{
                        clearTimeout(timeout);
                        target.innerHTML = '<div style="padding:10px;text-align:center;color:#666;font-size:12px;">Molecule: {smiles[:30]}...</div>';
                    }}
                }}
                
                loadAndRender();
            }})();
            </script>
            """
        st.components.v1.html(html, height=height+10)

    # Usage
    if st.session_state.get("smiles_input"):
        # Add scroll anchor for molecule visualization with header
        st.markdown('<div id="molecule-visualization"></div>', unsafe_allow_html=True)
        
        # Add a clear header when in detailed analysis mode
        if st.session_state.get("analyzed", False):
            st.subheader("🧬 Molecule Visualization & Results")
        
        # Use canonical SMILES for rendering if available, otherwise use input
        smiles_to_render = st.session_state.get("smiles") or st.session_state.smiles_input
        render_smiles_rdkitjs(smiles_to_render)
    else:
        st.warning("Enter a SMILES string above.")

    # -----------------------------
    # Run predictions with memory management
    # -----------------------------
    # Check for predict button clicks from both locations
    predict_clicked = st.session_state.get("detailed_predict_clicked", False) or st.session_state.get("regular_predict_clicked", False)
    
    # Reset the predict flags after checking
    if st.session_state.get("detailed_predict_clicked", False):
        st.session_state.detailed_predict_clicked = False
    if st.session_state.get("regular_predict_clicked", False):
        st.session_state.regular_predict_clicked = False
    
    if predict_clicked and (st.session_state.smiles_input and st.session_state.smiles_input.strip()):
        # Automatically switch to Binary tab when running predictions
        st.session_state.active_tab = "Binary"
        st.session_state.tab_selector = "🎯 Binary Classification"  # Update radio button state to new format
        
        # Reset scroll tracking for new prediction - ensure it works for both button types
        st.session_state["scroll_molecule_done"] = False
        
        # Ensure scroll will happen regardless of which predict button was clicked
        if st.session_state.get("detailed_predict_clicked", False) or st.session_state.get("regular_predict_clicked", False):
            st.session_state["scroll_to_molecule_requested"] = True
        
        # Synchronize the canonical SMILES for models from the top input
        smiles_to_predict = st.session_state.smiles_input.strip()
        
        # Add a loading indicator
        with st.spinner("Running predictions..."):
            try:
                # Clean up previous results to free memory
                cleanup_session_state()
                
                # Canonicalize the SMILES and store both versions
                mol = Chem.MolFromSmiles(smiles_to_predict)
                if mol is None:
                    raise ValueError("Invalid SMILES")
                
                canonical_smiles = Chem.MolToSmiles(mol)
                st.session_state.smiles = canonical_smiles  # For models
                st.session_state.mol = mol

                # Binary prediction with error handling
                try:
                    x_bin = featurize_fp_desc(mol)
                    proba = float(binary_model.predict_proba([x_bin])[0, 1])
                    st.session_state.binary_proba = proba
                    st.session_state.binary_label = int(proba >= st.session_state.bin_thr)
                except Exception as e:
                    st.error(f"Binary prediction failed: {e}")
                    st.session_state.binary_proba = None

                # Multi-task prediction with error handling
                try:
                    x_mt = featurize_fp_desc(mol)
                    st.session_state.x_mt = x_mt
                    probs = {}
                    for task in MT_TASKS:
                        try:
                            mdl = mt_models[task]
                            probs[task] = float(mdl.predict_proba([x_mt])[0, 1])
                        except Exception as task_error:
                            st.warning(f"Task {task} failed: {task_error}")
                            probs[task] = 0.0
                    st.session_state.mt_probs = probs
                except Exception as e:
                    st.error(f"Multi-task prediction failed: {e}")
                    st.session_state.mt_probs = None

                st.session_state.last_error = None
                
                # Store success messages in session state to persist across reruns
                st.session_state.prediction_success_message = f"✅ Predictions completed successfully for: **{smiles_to_predict}**"
                
                # Set scroll to molecule after successful prediction
                st.session_state["scroll_to_molecule"] = True
                
                if canonical_smiles != smiles_to_predict:
                    st.session_state.canonicalization_message = f"📝 Canonicalized as: **{canonical_smiles}**"
                else:
                    st.session_state.canonicalization_message = None
                
                # Set flag to refresh UI after prediction completes
                st.session_state.prediction_completed = True
                
            except Exception as e:
                st.session_state.last_error = str(e)
                st.error(f"❌ Prediction error: {e}")
                # Clear success messages on error
                st.session_state.prediction_success_message = None
                st.session_state.canonicalization_message = None
                # Clean up on error
                cleanup_session_state()

    # Display persistent success messages with smaller font
    if st.session_state.get("prediction_success_message"):
        st.markdown(f"""
        <div style="
            background-color: #d4edda; 
            border: 1px solid #c3e6cb; 
            border-radius: 0.25rem; 
            padding: 0.5rem 0.75rem; 
            margin-bottom: 1rem;
            font-size: 0.8rem;
            color: #155724;
        ">
            {st.session_state.prediction_success_message}
        </div>
        """, unsafe_allow_html=True)
    # Removed canonicalization message display

    # Handle UI refresh after prediction completes
    if st.session_state.get("prediction_completed", False):
        st.session_state.prediction_completed = False
        st.rerun()

    if st.session_state.last_error:
        st.error(f"Prediction error: {st.session_state.last_error}")

    # Show a helpful message if no SMILES is entered but predict is clicked
    if predict_clicked and not (st.session_state.smiles_input and st.session_state.smiles_input.strip()):
        st.error("⚠️ Please enter a SMILES string in the input box above before clicking Predict.")


    # --- Tab selector (radio replaces tabs) with enhanced visual styling
    tab_options = [
        "🎯 Binary Classification", 
        "🔬 Multi-Task Analysis", 
        "💡 Explainability",
        "⚗️ ADMET Profile",
        "📋 Summary"
    ]
    tab_keys = ["Binary", "Multi-Task", "Explain", "ADMET", "Brief Summary"]
    
    # Add custom CSS for better radio button styling
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > div {
        gap: 1.5rem !important;
    }
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        border: 2px solid #e9ecef !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 2px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        color: #495057 !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: transparent !important;
        border-color: #0066cc !important;
        color: #0066cc !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:checked + div {
        background-color: transparent !important;
        color: #0066cc !important;
        border-color: #0066cc !important;
        border-width: 2px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    selected_tab = st.radio(
        "📊 Analysis View",
        options=tab_options,
        index=tab_keys.index(st.session_state.active_tab) if st.session_state.active_tab in tab_keys else 0,
        horizontal=True,
        key="tab_selector"
    )
    
    # Convert back to internal key
    tab = tab_keys[tab_options.index(selected_tab)]


    # --- Render sections based on selected tab
    if tab == "Binary":
        proba = st.session_state.binary_proba
        if proba is None:
            st.info("Enter a SMILES and click Predict.")
        else:
            st.metric("Probability (toxic = 1)", f"{proba:.3f}")
            if st.session_state.binary_label == 1:
                st.error("Prediction: **Toxic**")
            else:
                st.success("Prediction: **Non-toxic**")
            st.caption("Adjust the threshold above to match risk tolerance.")
        
        # Add consistent spacing to match other tabs
        st.markdown("<br>" * 2, unsafe_allow_html=True)


    elif tab == "Multi-Task":
        probs = st.session_state.mt_probs
        if probs is None:
            st.info("Enter a SMILES and click Predict.")
            # Add consistent spacing when no data
            st.markdown("<br>" * 2, unsafe_allow_html=True)
        else:
            df = (pd.DataFrame.from_dict(probs, orient="index", columns=["probability"])
                    .reset_index().rename(columns={"index": "task"}))
            df["risk"] = (df["probability"] >= st.session_state.mt_thr).astype(int)
            df["family"] = df["task"].str.split("-", n=1).str[0]
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)


            st.subheader("Tox21 pathway probabilities")
            st.dataframe(df.style.format({"probability": "{:.3f}"}), use_container_width=True)

            import altair as alt
            def chart_for(subdf, title):
                if subdf.empty:
                    st.info(f"No {title} tasks.")
                    return
                c = alt.Chart(subdf).mark_bar().encode(
                    x=alt.X('probability:Q', title='Probability'),
                    y=alt.Y('task:N', sort='-x', title=None),
                    color=alt.condition(
                        alt.datum.probability >= st.session_state.mt_thr,
                        alt.value('#d62728'), alt.value('#1f77b4')
                    ),
                    tooltip=['task','probability']
                ).properties(height=max(200, 22*len(subdf)), title=title)
                st.altair_chart(c, use_container_width=True)

            col_nr, col_sr = st.columns(2)
            with col_nr:
                chart_for(df[df["family"]=="NR"], "NR family")
            with col_sr:
                chart_for(df[df["family"]=="SR"], "SR family")


    elif tab == "Explain":
        if st.session_state.mt_probs is None or st.session_state.x_mt is None or st.session_state.mol is None:
            st.info("Run Predict first, then choose a pathway to explain.")
        else:
            exp_task = st.selectbox("Pick a pathway", MT_TASKS, index=0, key="exp_task")
            top_k = st.slider("Top substructures to highlight", 3, 12, 6, 1, key="exp_topk")
            run_explain = st.button("Run SHAP + Atom Highlights", key="btn_explain")

            if run_explain:
                try:
                    import numpy as np
                    import shap

                    mdl = mt_models[exp_task]

                    # Make sure the input is 2D: (1, n_features)
                    X_sample = np.asarray(st.session_state.x_mt, dtype=float).reshape(1, -1)

                    # TreeExplainer works directly on GradientBoostingClassifier
                    explainer = shap.TreeExplainer(mdl)

                    # Compute SHAP values
                    sv = explainer.shap_values(X_sample)  # can be np.ndarray or list (multiclass)

                    # If SHAP returns a list (e.g., one array per class), pick the positive class
                    if isinstance(sv, list):
                        # class 1 = “toxic” (positive class)
                        sv = sv[1]

                    # sv now has shape (1, n_features); take the first row
                    shap_row = sv[0]

                    # (Optional) descriptor table
                    desc_names = ["MolWt","LogP","HAcc","HDon","TPSA"]
                    desc_contrib = (
                        pd.DataFrame({
                            "feature": [f"fp_{i}" for i in range(FP_BITS)] + desc_names,
                            "shap": shap_row
                        })
                        .tail(5)
                        .sort_values("shap", key=np.abs, ascending=False)
                        .reset_index(drop=True)
                    )
                    st.session_state.desc_table = desc_contrib

                    # Atom-level highlight image from top fingerprint bits
                    payload = draw_payload_with_shap(st.session_state.smiles, shap_row, top_k=6, radius=2)

                    if payload:
                        render_shap_rdkitjs(payload, size=(600, 420))
                    else:
                        st.error("Invalid SMILES.")

                    
                except Exception as e:
                    st.session_state.last_error = str(e)


            # Show results if available
            if st.session_state.get("desc_table") is not None:
                st.write(f"**{st.session_state.get('exp_task') or exp_task}** probability: "
                        f"{st.session_state.get('mt_probs', {}).get(st.session_state.get('exp_task') or exp_task, float('nan')):.3f}")
                st.write("Descriptor contributions (|SHAP|):")
                
                # Display as compact table without stretching or scrolling
                st.dataframe(
                    st.session_state.desc_table.style.format({"shap": "{:.4f}"}),
                    use_container_width=False,
                    width=400
                )
            if st.session_state.get("explain_img") is not None:
                st.image(st.session_state.explain_img, caption="Top substructures by |SHAP|", width='stretch')

    elif tab == "Brief Summary":

        mol = st.session_state.get("mol")
        if mol is None:
            st.error("⚠️ No molecular data available. Please run prediction first.")
        else:
            desc = compute_descriptors(mol)
            bin_proba = float(st.session_state.get("binary_proba", 0.0))
            mt_probs = st.session_state.get("mt_probs", {})

            # Properties table (key–value)
            df_props = pd.DataFrame(list(desc.items()), columns=["Property", "Value"])
            df_props["Value"] = df_props["Value"].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float, float)) else v)

            c1, c2 = st.columns([1,1.4])
            with c1:
                st.subheader("Properties")
                st.dataframe(df_props, use_container_width=True)

            with c2:
                st.subheader("Chemist-friendly summary")
                bullets = copilot_bullets(desc, mt_probs, bin_proba, bin_thr=st.session_state.get("bin_thr", 0.5))
                st.markdown("\n".join([f"- {b}" for b in bullets]))

    elif tab == "ADMET":
        # Single-column layout (left-only). Users scroll to see details.
        st.header("ADMET — Predictions, Assessments & Provenance")

        project_type = st.radio(
            "Indication setting (affects BBBP desirability)",
            ["Non-CNS", "CNS"],
            horizontal=True,
            key="admet_proj"
        )

        smiles = st.session_state.get("smiles", "")
        if not smiles:
            st.info("Enter a SMILES string (top of the app) to get ADMET predictions.")
        else:
            preds = predict_admet(smiles)
            if preds is None:
                st.error("Invalid SMILES.")
            else:
                # Build rows for the visible table. Each row will include:
                # Endpoint, Value (raw), Type (reg/prob), Assessment, Source (where Value came from)
                rows = []
                for endpoint in ["esol", "lipo", "bbbp"]:
                    p = preds.get(endpoint)
                    if p is None:
                        continue
                    raw_val = p["value"]
                    #mtype = p.get("type", "regression")
                    #task = p.get("task", "")
                    # Interpretation — keep using your helper interpreters
                    if endpoint == "esol":
                        assessment = interpret_logS(raw_val)
                        name = "Solubility (logS, ESOL)"
                        value_display = f"{raw_val:.2f}"
                        #source = "Model predict() → regression output (logS)"
                    elif endpoint == "lipo":
                        assessment = interpret_logD(raw_val)
                        name = "Lipophilicity (logD)"
                        value_display = f"{raw_val:.2f}"
                        #source = "Model predict() → regression output (logD)"
                    elif endpoint == "bbbp":
                        assessment = interpret_bbbp(raw_val, project=project_type)
                        name = "BBBP (probability)"
                        value_display = f"{raw_val:.2f}"
                        #source = "Model predict_proba() → P(BBBP=1)"
                    else:
                        assessment = "—"
                        name = endpoint
                        value_display = str(raw_val)
                        #source = "Model output"

                    rows.append({
                        "Endpoint": name,
                        "Value (ML Model Prediction)": value_display,
                        #"Type": mtype,
                        "Assessment": assessment,
                        #"Value source": source,
                        #"Model task (manifest)": task
                    })

                # Show the compact table (left column single flow)
                st.subheader("Predicted ADME endpoints (raw + assessment)")
                df_admet = pd.DataFrame(rows)
                # Make the table easy to skim
                #st.dataframe(df_admet, use_container_width=True, height=220)
                st.dataframe(df_admet)

                # --- Developability score and breakdown (explicit provenance) ---
                st.subheader("Developability score & breakdown")

                # Allow user to choose whether to invert BBBP for Non-CNS automatically
                st.caption("Defaults: sol 0.4 | lipo 0.4 | bbbp 0.2 (weights are customizable)")
                weights = {"sol": 0.4, "lipo": 0.4, "bbbp": 0.2}
                
                raw_logS = preds.get("esol", {}).get("value", np.nan)
                raw_logD = preds.get("lipo", {}).get("value", np.nan)
                raw_bbbp = preds.get("bbbp", {}).get("value", np.nan)

                sol_score = np.nan
                lipo_score = np.nan
                bbbp_score = np.nan

                if not np.isnan(raw_logS):
                    sol_score = float(np.clip((raw_logS + 6.0) / 6.0, 0.0, 1.0))
                if not np.isnan(raw_logD):
                    lipo_score = float(1.0 - np.clip(abs(raw_logD - 2.0) / 3.0, 0.0, 1.0))
                if not np.isnan(raw_bbbp):
                    # default behavior in your function: invert for Non-CNS
                    if project_type == "CNS":
                        bbbp_score = float(np.clip(raw_bbbp, 0.0, 1.0))
                    else:
                        bbbp_score = float(np.clip(1.0 - raw_bbbp, 0.0, 1.0))

                # Normalize weights for only present components
                comps = {}
                present = {}
                if not np.isnan(sol_score): present["sol"] = sol_score
                if not np.isnan(lipo_score): present["lipo"] = lipo_score
                if not np.isnan(bbbp_score): present["bbbp"] = bbbp_score

                total_w = sum([weights[k] for k in present.keys()]) if present else 1.0
                composite = 0.0
                for k, v in present.items():
                    norm_w = (weights[k] / total_w) if total_w > 0 else 0.0
                    contrib = v * norm_w
                    comps[k] = {"subscore": v, "normalized_weight": norm_w, "contribution": contrib}
                    composite += contrib

                composite = float(np.clip(composite, 0.0, 1.0))

                # Show composite metric and color
                def score_badge(s):
                    if s >= 0.8: return "✅ High"
                    if s >= 0.6: return "🟡 Moderate"
                    return "🔴 Low"

                st.metric(label=f"Developability composite (0–1) — {score_badge(composite)}", value=f"{composite:.2f}")

                # Breakdown table (explicit numbers for audit)
                if comps:
                    breakdown_rows = []
                    name_map = {"sol": "Solubility", "lipo": "Lipophilicity", "bbbp": "BBBP (proj adj)"}
                    for k, v in comps.items():
                        breakdown_rows.append({
                            "Component": name_map[k],
                            "Subscore (0–1)": f"{v['subscore']:.3f}",
                            "Normalized weight": f"{v['normalized_weight']:.3f}",
                            "Contribution": f"{v['contribution']:.3f}"
                        })
                    df_break = pd.DataFrame(breakdown_rows)
                    st.table(df_break)

                    # small bar chart of contributions
                    chart_df = pd.DataFrame({
                        "component": [r["Component"] for r in breakdown_rows],
                        "contribution": [float(r["Contribution"]) for r in breakdown_rows]
                    }).set_index("component")
                    st.markdown("**Contribution chart**")
                    st.bar_chart(chart_df)

                # Downloadable provenance CSV
                combined_rows = []
                for r in rows:
                    # attach normalized subscores & contributions if available
                    key = None
                    if r["Endpoint"].lower().startswith("solub"):
                        key = "sol"
                    elif r["Endpoint"].lower().startswith("lipo"):
                        key = "lipo"
                    elif r["Endpoint"].lower().startswith("bbbp"):
                        key = "bbbp"
                    combined = r.copy()
                    if key and key in comps:
                        c = comps[key]
                        combined.update({
                            "subscore": c["subscore"],
                            "norm_weight": c["normalized_weight"],
                            "contribution": c["contribution"]
                        })
                    combined_rows.append(combined)

                df_export = pd.DataFrame(combined_rows)
                st.download_button(
                    "Download ADMET + developability breakdown (CSV)",
                    data=df_export.to_csv(index=False),
                    file_name="admet_developability_breakdown.csv",
                    mime="text/csv"
                )

                # --- Provenance & Assumptions (explicit) ---
                st.divider()
                st.subheader("Assumptions & how the table columns are computed")
                with st.expander("Click to expand: full assumptions, thresholds, and where 'Value' comes from"):
                    st.markdown("### Where the **Value (raw)** column comes from")
                    st.markdown(
                        "- Each 'Value (raw)' is the model output from `predict_admet(smiles)`.\n"
                        "- For regression endpoints (ESOL, LIPO) the value shown is `model.predict()` → a continuous predicted value (e.g., logS, logD).\n"
                        "- For classification endpoints (BBBP) the value shown is `model.predict_proba()` → probability of BBBP=1 (range 0–1).\n"
                        "- The models were loaded from the manifest in the ADMET_DIR and were applied to features produced by `featurize_fp_desc(mol)` (FP + descriptors)."
                    )
                    st.markdown("### How the **Assessment** column is generated (exact rules)")
                    st.markdown(
                        "- **Solubility (ESOL, logS)**: \n"
                        "  - `> -3` → ✅ good\n"
                        "  - `-4 < = logS <= -3` → 🟡 moderate\n"
                        "  - `<= -4` → 🔴 poor\n"
                    )
                    st.markdown(
                        "- **Lipophilicity (logD)**: ideal ≈ 1–3\n"
                        "  - `1.0 <= logD <= 3.0` → ✅ in-range\n"
                        "  - `0.5 <= logD < 1.0` or `3.0 < logD <= 3.5` → 🟡 borderline\n"
                        "  - otherwise → 🔴 out-of-range\n"
                    )
                    st.markdown(
                        "- **BBBP (probability)**: interpretation depends on project\n"
                        "  - For **CNS** projects: `p >= 0.7` → ✅ high (desired), `0.4 <= p < 0.7` → 🟡 medium, `p < 0.4` → 🔴 low\n"
                        "  - For **Non-CNS** projects: `p <= 0.3` → ✅ low (desired), `0.3 < p <= 0.6` → 🟡 medium, `p > 0.6` → 🔴 high\n"
                    )
                    st.markdown("### How the developability composite is computed (exact formula)")
                    st.markdown(
                        "- Subscores:\n"
                        "  - `sol_score = clip((logS + 6) / 6, 0, 1)` — maps logS from [-6..0] → [0..1]\n"
                        "  - `lipo_score = 1 - clip(|logD - 2| / 3, 0, 1)` — peaks at logD=2 and falls off\n"
                        "  - `bbbp_score = p` for CNS; for Non-CNS we use `1 - p` (low BBBP desirable)\n"
                        "- Composite = sum(normalized_weight_i * subscore_i) where normalized_weight_i = weight_i / sum(weights_of_present_components)\n"
                    )
                    st.markdown("### Feature & model provenance")
                    st.markdown(
                        "- Models loaded from `ADMET_DIR/manifest.json` → model file paths were loaded with `joblib.load()`.\n"
                        "- Features are created by `featurize_fp_desc(mol)` (FP + descriptors). Feature configuration loaded from manifest: shown below."
                    )
                    # show manifest-derived feature config to be fully transparent
                    st.json(admet_featcfg)
else:
    st.caption("Tip: Paste a SMILES above and click Predict/Analyze — the Copilot will auto-run the models and unlock detailed tabs.")

# Handle scroll behaviors based on user actions
# Check if we just completed a prediction and should scroll to molecule
if (st.session_state.get("scroll_to_molecule", False) or 
    st.session_state.get("scroll_to_molecule_requested", False) or
    (st.session_state.get("prediction_success_message") and 
     not st.session_state.get("scroll_molecule_done", False))):
    
    st.components.v1.html("""
    <script>
        setTimeout(function() {
            console.log('Attempting to scroll to molecule visualization...');
            var target = document.getElementById('molecule-visualization');
            if (target) {
                // Scroll to molecule visualization with more prominent positioning
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                    inline: 'nearest'
                });
                // Add a slight delay then focus to ensure it's visible
                setTimeout(function() {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    console.log('Successfully scrolled to molecule visualization');
                }, 100);
            } else {
                console.log('Molecule visualization anchor not found');
            }
        }, 300);
    </script>
    """, height=0)
    # Clear all scroll flags and mark as done
    st.session_state["scroll_to_molecule"] = False
    st.session_state["scroll_to_molecule_requested"] = False
    st.session_state["scroll_molecule_done"] = True

# Handle scroll to SMILES input when detailed analysis is triggered
if st.session_state.get("scroll_to_smiles_input", False):
    st.components.v1.html("""
    <script>
        setTimeout(function() {
            var target = document.getElementById('smiles-input-section');
            if (target) {
                // Get the target position and add larger offset to prevent cutting off
                var targetRect = target.getBoundingClientRect();
                var offset = 120; // Increased to 120px offset from top for better visibility
                var scrollPosition = window.pageYOffset + targetRect.top - offset;
                
                // Ensure we don't scroll above the page
                scrollPosition = Math.max(0, scrollPosition);
                
                // Smooth scroll to position with offset
                window.scrollTo({
                    top: scrollPosition,
                    behavior: 'smooth'
                });
                
                // Double-check scroll after additional delay for first-time analysis
                setTimeout(function() {
                    var targetRect2 = target.getBoundingClientRect();
                    var scrollPosition2 = window.pageYOffset + targetRect2.top - offset;
                    scrollPosition2 = Math.max(0, scrollPosition2);
                    window.scrollTo({
                        top: scrollPosition2,
                        behavior: 'smooth'
                    });
                }, 200);
            } else {
                console.log('SMILES input anchor not found, scrolling to top with offset');
                window.scrollTo({
                    top: 80,
                    behavior: 'smooth'
                });
            }
        }, 500);
    </script>
    """, height=0)
    # Clear the scroll flag
    st.session_state["scroll_to_smiles_input"] = False

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Your personal co-pilot for early drug discovery — enter a molecule, and instantly explore its toxicity, ADMET profile, and developability score.")
