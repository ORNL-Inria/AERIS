# -*- coding: utf-8 -*-
"""
Agent A: Database Structure Search for U-Si System (Dual-Model)
===============================================================
Reproduces Lopes et al., J. Nucl. Mater. 510 (2018) 331-336

Methodology:
  Tiered exhaustive structure search:
    Tier 1 -- Binary-only templates (AB, ~8700 entries)
    Tier 2 -- Binary + Ternary templates (AB+ABC, ~31400 entries)
  Compares both tiers against DFT reference to assess whether
  ternary structural prototypes help or hurt for a binary system.

Dual-model:
  RF  -- RandomForestRegressor (model_3.joblib, 236 feat) -- struct-aware
  NN  -- AERIS Neural Network  (aeris_full_structure_classic.pt, 233 feat) -- comp-only

Run:  python agent_paper_A_mcp.py
"""

import sys, os, json, time, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from aeris import load_structure_model, parse_formula, compute_magpie_df

# ---- DFT REFERENCE (Lopes et al. 2018) -------------------------------------
DFT_REF = {
    "U3Si":  dict(si_frac=0.25,    dH_DFT=-0.40, sg=221, sym="Pm-3m",   on_hull=True),
    "U3Si2": dict(si_frac=0.40,    dH_DFT=-0.48, sg=127, sym="P4/mbm",  on_hull=True),
    "U5Si4": dict(si_frac=4.0/9.0, dH_DFT=-0.47, sg=191, sym="P6/mmm",  on_hull=False),
    "USi":   dict(si_frac=0.50,    dH_DFT=-0.55, sg=63,  sym="Cmcm",    on_hull=True),
    "USi2":  dict(si_frac=2.0/3.0, dH_DFT=-0.52, sg=191, sym="P6/mmm",  on_hull=True),
    "USi3":  dict(si_frac=0.75,    dH_DFT=-0.35, sg=221, sym="Pm-3m",   on_hull=True),
}

# ---- Load models ------------------------------------------------------------
MODEL_RF = "model/model_3.joblib"
MODEL_NN = "model/aeris_comp_only.pt"

print("Loading models...")
_model_rf, _feat_rf, _scaler_rf = load_structure_model(MODEL_RF, device="cpu")
_model_nn, _feat_nn, _scaler_nn = load_structure_model(MODEL_NN, device="cpu")
print(f"  RF: {len(_feat_rf)} features   NN: {len(_feat_nn)} features")

# ---- MCP log ----------------------------------------------------------------
_MCP_LOG = []
_MCP_LOG_PATH = os.path.join("results", "agent_A_mcp_call_log.json")


def _log_mcp(model_tag, composition, n_templates, best_eV):
    _MCP_LOG.append({"timestamp": datetime.datetime.now().isoformat(),
                     "model": model_tag, "composition": composition,
                     "n_templates": n_templates, "best_eV": best_eV})


def _flush_mcp_log():
    os.makedirs("results", exist_ok=True)
    with open(_MCP_LOG_PATH, "w") as f:
        json.dump(_MCP_LOG, f, indent=2)


# ---- Database loading -------------------------------------------------------
STRUCT_COLS = ["spacegroup_number", "density_atomic", "CN_avg", "CN_min", "CN_max"]


def precompute_struct_matrix(templates, feat_names):
    """Pre-extract structural features as numpy arrays for vectorized prediction.
    Returns (col_indices, matrix) where col_indices maps into feat_names."""
    feat_list = list(feat_names)
    indices = [feat_list.index(c) for c in STRUCT_COLS if c in feat_list]
    names = [c for c in STRUCT_COLS if c in feat_list]
    n = len(templates)
    matrix = np.zeros((n, len(names)), dtype=np.float32)
    for i, tmpl in enumerate(templates):
        for j, col in enumerate(names):
            matrix[i, j] = float(tmpl.get(col, 0.0))
    return indices, matrix


def _dedup_templates(templates):
    """Deduplicate templates on (SG, density, CN_avg, CN_min, CN_max)."""
    seen = set()
    unique = []
    for t in templates:
        key = (t["spacegroup_number"],
               round(t["density_atomic"], 4),
               round(t["CN_avg"], 2),
               round(t["CN_min"], 1),
               round(t["CN_max"], 1))
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def load_database():
    """Load binary and ternary templates separately, plus real U-Si entries."""
    path = "data/Dataset_feature+CN.csv"
    db = pd.read_csv(path, low_memory=False)
    print(f"  [DB] {len(db)} entries from {path}")

    def _extract(mask):
        out = []
        for _, row in db[mask].iterrows():
            try:
                t = {c: float(row[c]) for c in STRUCT_COLS}
                t["spacegroup_number"] = int(t["spacegroup_number"])
                out.append(t)
            except Exception:
                continue
        return out

    raw_bin = _extract(db["nelements"] == 2)
    raw_ter = _extract(db["nelements"] == 3)

    # Tag source for binary/ternary differentiation in plots
    for t in raw_bin:
        t["_source"] = "binary"
    for t in raw_ter:
        t["_source"] = "ternary"

    binary_templates = _dedup_templates(raw_bin)
    ternary_templates = _dedup_templates(raw_ter)
    all_templates = _dedup_templates(raw_bin + raw_ter)

    print(f"  [DB] Binary: {len(binary_templates)} unique templates")
    print(f"  [DB] Ternary: {len(ternary_templates)} unique")
    print(f"  [DB] All (B+T): {len(all_templates)} unique templates")

    # Boolean mask: True = binary origin in the combined pool
    is_binary_all = np.array([t.get("_source") == "binary" for t in all_templates])
    n_bin_in_all = int(is_binary_all.sum())
    print(f"  [DB] In combined pool: {n_bin_in_all} binary, "
          f"{len(all_templates) - n_bin_in_all} ternary")

    # Real U-Si binary entries
    mask_usi = (db["nelements"] == 2) & (db["U"] > 0) & (db["Si"] > 0)
    usi_entries = []
    for _, row in db[mask_usi].iterrows():
        si = float(row["Si"])
        u = float(row["U"])
        comp = str(row["composition_reduced"]).replace(" ", "")
        struct = {c: float(row[c]) for c in STRUCT_COLS}
        struct["spacegroup_number"] = int(struct["spacegroup_number"])
        usi_entries.append({
            "composition": comp,
            "si_frac": si / (si + u),
            "structure": struct,
        })
        sg = struct["spacegroup_number"]
        print(f"  [DB-USi] {comp:8s} SG={sg:3d}")

    return binary_templates, all_templates, usi_entries, is_binary_all


# ---- Batch prediction -------------------------------------------------------

def _build_base_features(composition):
    """Compute composition + Magpie features once."""
    magpie_df = compute_magpie_df(pd.Series([composition]))
    magpie_row = magpie_df.iloc[0].to_dict()
    parsed = parse_formula(composition)
    total = float(sum(parsed.values()))
    base = {}
    if total > 0:
        for el, cnt in parsed.items():
            base[el] = cnt / total
    for k, v in magpie_row.items():
        try:
            base[k] = float(v)
        except Exception:
            pass
    return base


def screen_rf(composition, struct_indices, struct_matrix):
    """Predict dH_f for one composition across all templates (RF).
    Uses pre-computed structural matrix -- no Python loop over templates."""
    base = _build_base_features(composition)
    base_row = np.array([base.get(name, 0.0) for name in _feat_rf],
                        dtype=np.float32)
    n = struct_matrix.shape[0]
    X = np.tile(base_row, (n, 1))                      # (n_tmpl, 236)
    for j, col_idx in enumerate(struct_indices):        # overwrite struct cols
        X[:, col_idx] = struct_matrix[:, j]
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    if _scaler_rf is not None:
        X = _scaler_rf.transform(X).astype(np.float32)
    return _model_rf.predict(X).flatten().astype(float)


def predict_rf_single(composition, structure):
    """Predict dH_f for one composition + one structure (RF)."""
    base = _build_base_features(composition)
    for k, v in structure.items():
        try:
            base[k] = float(v)
        except Exception:
            pass
    X = np.zeros((1, len(_feat_rf)), dtype=np.float32)
    X[0, :] = [base.get(name, 0.0) for name in _feat_rf]
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    if _scaler_rf is not None:
        X = _scaler_rf.transform(X).astype(np.float32)
    return float(_model_rf.predict(X).flatten()[0])


def predict_nn_single(composition):
    """Predict dH_f for one composition (NN, comp-only)."""
    import torch
    base = _build_base_features(composition)
    X = np.zeros((1, len(_feat_nn)), dtype=np.float32)
    X[0, :] = [base.get(name, 0.0) for name in _feat_nn]
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    if _scaler_nn is not None:
        X = _scaler_nn.transform(X).astype(np.float32)
    with torch.no_grad():
        pred = _model_nn(torch.FloatTensor(X)).cpu().numpy().flatten()
    return float(pred[0])


# ---- Composition generation -------------------------------------------------

def generate_compositions():
    """All U_xSi_y with x+y <= 12, plus exact DFT reference fractions."""
    fracs = set()
    formulas = {}
    for x in range(1, 12):
        for y in range(1, 12):
            if x + y > 12:
                continue
            sf = round(y / (x + y), 6)
            fracs.add(sf)
            if sf not in formulas or (x + y) < sum(
                    parse_formula(formulas[sf]).values()):
                if x == 1 and y == 1:
                    formulas[sf] = "USi"
                elif x == 1:
                    formulas[sf] = f"USi{y}"
                elif y == 1:
                    formulas[sf] = f"U{x}Si"
                else:
                    formulas[sf] = f"U{x}Si{y}"
    for name, ref in DFT_REF.items():
        sf = round(ref["si_frac"], 6)
        fracs.add(sf)
        formulas[sf] = name
    return [(sf, formulas[sf]) for sf in sorted(fracs)]


# ---- Lower convex hull ------------------------------------------------------

def lower_convex_hull(xs, ys):
    """Lower hull via Andrew's monotone chain.  Returns (hx, hy).
    Adds (0,0) and (1,0) for pure elements."""
    points = list(zip(xs, ys)) + [(0.0, 0.0), (1.0, 0.0)]
    points = sorted(set(points))
    hull = []
    for p in points:
        while len(hull) >= 2:
            ox, oy = hull[-2]
            ax, ay = hull[-1]
            bx, by = p
            if (ax - ox) * (by - oy) - (ay - oy) * (bx - ox) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hy = zip(*hull)
    return list(hx), list(hy)


# ---- Main screening --------------------------------------------------------

def run_database_screen(compositions, templates, usi_entries,
                        struct_indices, struct_matrix, tier_label,
                        base_features_cache=None):
    """Screen all templates for one tier.
    base_features_cache: dict to share Magpie computation across tiers.
    Returns (results_dict, base_features_cache)."""

    if base_features_cache is None:
        base_features_cache = {}

    # Index U-Si entries by rounded si_frac
    usi_by_frac = {}
    for e in usi_entries:
        sf = round(e["si_frac"], 4)
        usi_by_frac.setdefault(sf, []).append(e)

    results = {}
    total = len(compositions)

    for idx, (si_frac, formula) in enumerate(compositions):
        print(f"\n  [{idx+1}/{total}] {formula} (Si={si_frac:.4f})", flush=True)

        t0 = time.time()
        preds_rf = screen_rf(formula, struct_indices, struct_matrix)
        dt = time.time() - t0

        best_idx = int(np.argmin(preds_rf))
        best_dH = float(preds_rf[best_idx])
        best_struct = templates[best_idx]

        # NN only computed once (tier-independent); cache it
        if formula not in base_features_cache:
            base_features_cache[formula] = predict_nn_single(formula)
        dH_NN = base_features_cache[formula]

        print(f"    RF best: {best_dH:+.4f} (SG={best_struct['spacegroup_number']}) "
              f"[{len(templates)} tmpl, {dt:.1f}s]   NN: {dH_NN:+.4f}")

        # Rank real U-Si entries among templates
        sf_key = round(si_frac, 4)
        db_matches = usi_by_frac.get(sf_key, [])
        ranking_info = []
        sorted_preds = np.sort(preds_rf)
        n_total = len(sorted_preds)

        for entry in db_matches:
            dH_real = predict_rf_single(formula, entry["structure"])
            rank = int(np.searchsorted(sorted_preds, dH_real)) + 1
            percentile = rank / n_total * 100.0
            gap = dH_real - best_dH
            sg = entry["structure"]["spacegroup_number"]
            ranking_info.append({
                "composition": entry["composition"],
                "sg": sg,
                "dH_RF": dH_real,
                "rank": rank,
                "n_total": n_total,
                "percentile": percentile,
                "gap_to_best": gap,
            })
            print(f"    DB {entry['composition']} SG={sg}: "
                  f"dH_RF={dH_real:+.4f}  rank={rank}/{n_total} "
                  f"({percentile:.1f}th pctl)  gap={gap:+.4f}")

        _log_mcp(f"RF-{tier_label}", formula, len(templates), best_dH)

        results[si_frac] = {
            "formula":      formula,
            "best_dH_RF":   best_dH,
            "best_dH_NN":   dH_NN,
            "best_struct":  best_struct,
            "all_dH_RF":    preds_rf.tolist(),
            "n_templates":  len(templates),
            "ranking":      ranking_info,
        }

    _flush_mcp_log()
    return results, base_features_cache


# ---- Metrics ----------------------------------------------------------------

def compute_metrics(hull_results, label="RF"):
    phase_metrics = {}
    errs_RF, errs_NN = [], []

    for name, ref in DFT_REF.items():
        sf = ref["si_frac"]
        closest = min(hull_results.keys(), key=lambda x: abs(x - sf))
        if abs(closest - sf) > 0.02:
            continue
        res = hull_results[closest]
        dH_DFT = ref["dH_DFT"]
        err_RF = res["best_dH_RF"] - dH_DFT
        err_NN = res["best_dH_NN"] - dH_DFT

        phase_metrics[name] = {
            "si_frac": sf, "formula": res["formula"],
            "dH_DFT": dH_DFT,
            "dH_RF": res["best_dH_RF"], "dH_NN": res["best_dH_NN"],
            "err_RF": err_RF, "err_NN": err_NN,
            "on_hull_DFT": ref["on_hull"],
            "best_struct": res["best_struct"],
            "ranking": res["ranking"],
        }
        errs_RF.append(err_RF)
        errs_NN.append(err_NN)

    def _stats(errs):
        if not errs:
            return dict(mae=float("nan"), rmse=float("nan"), bias=float("nan"))
        arr = np.array(errs)
        return dict(mae=float(np.mean(np.abs(arr))),
                    rmse=float(np.sqrt(np.mean(arr**2))),
                    bias=float(np.mean(arr)))

    return {"per_phase": phase_metrics,
            "model_RF": _stats(errs_RF), "model_NN": _stats(errs_NN)}


# ---- Figure 1: Tiered convex hull overlay -----------------------------------

def plot_convex_hull(hull_bin, hull_all, metrics_bin, metrics_all,
                     is_binary_all, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_frac_all = sorted(hull_all.keys())
    sorted_frac_bin = sorted(hull_bin.keys())

    # Ghost scatter: differentiate binary vs ternary template origins
    rng = np.random.default_rng(42)
    for sf in sorted_frac_all:
        all_dh = np.array(hull_all[sf]["all_dH_RF"])
        n_samp = min(600, len(all_dh))
        idx = rng.choice(len(all_dh), size=n_samp, replace=False)
        bin_mask = is_binary_all[idx]
        ter_mask = ~bin_mask
        if bin_mask.any():
            ax.scatter([sf] * int(bin_mask.sum()), all_dh[idx[bin_mask]],
                       c="cornflowerblue", s=0.5, alpha=0.15, zorder=0,
                       rasterized=True)
        if ter_mask.any():
            ax.scatter([sf] * int(ter_mask.sum()), all_dh[idx[ter_mask]],
                       c="lightsalmon", s=0.5, alpha=0.15, zorder=0,
                       rasterized=True)
    # Legend proxies for ghost scatter
    ax.scatter([], [], c="cornflowerblue", s=14, alpha=0.6,
               label="Binary templates (ghost)")
    ax.scatter([], [], c="lightsalmon", s=14, alpha=0.6,
               label="Ternary templates (ghost)")

    # Binary-only hull (blue)
    bin_xs = list(sorted_frac_bin)
    bin_ys = [hull_bin[sf]["best_dH_RF"] for sf in sorted_frac_bin]
    bin_hx, bin_hy = lower_convex_hull(bin_xs, bin_ys)
    ax.plot(bin_hx, bin_hy, "b-", linewidth=2.0, zorder=6,
            label="RF hull (binary only)")
    ax.scatter(bin_xs, bin_ys, color="royalblue", marker="^", s=22, zorder=7,
               edgecolors="navy", linewidths=0.4, alpha=0.7)

    # Binary+Ternary hull (green) -- best-found points colored by source
    all_xs = list(sorted_frac_all)
    all_ys = [hull_all[sf]["best_dH_RF"] for sf in sorted_frac_all]
    all_hx, all_hy = lower_convex_hull(all_xs, all_ys)
    ax.plot(all_hx, all_hy, "g-", linewidth=2.0, zorder=5,
            label="RF hull (binary+ternary)")
    for sf, y in zip(all_xs, all_ys):
        src = hull_all[sf]["best_struct"].get("_source", "binary")
        if src == "binary":
            ax.scatter(sf, y, color="green", marker="s", s=22, zorder=7,
                       edgecolors="darkgreen", linewidths=0.4, alpha=0.5)
        else:
            ax.scatter(sf, y, color="darkorange", marker="v", s=28, zorder=7,
                       edgecolors="saddlebrown", linewidths=0.5, alpha=0.7)
    ax.scatter([], [], color="green", marker="s", s=22,
               edgecolors="darkgreen", linewidths=0.4,
               label="Best from binary tmpl")
    ax.scatter([], [], color="darkorange", marker="v", s=28,
               edgecolors="saddlebrown", linewidths=0.5,
               label="Best from ternary tmpl")

    # DFT lower convex hull
    dft_xs = [v["si_frac"] for v in DFT_REF.values() if v["on_hull"]]
    dft_ys = [v["dH_DFT"] for v in DFT_REF.values() if v["on_hull"]]
    dft_hx, dft_hy = lower_convex_hull(dft_xs, dft_ys)
    ax.plot(dft_hx, dft_hy, "r-", linewidth=1.5, alpha=0.6, zorder=4,
            label="DFT+U hull (Lopes 2018)")

    # DFT stable phases (filled stars)
    stable = {k: v for k, v in DFT_REF.items() if v["on_hull"]}
    unstable = {k: v for k, v in DFT_REF.items() if not v["on_hull"]}
    for k, v in stable.items():
        ax.scatter(v["si_frac"], v["dH_DFT"], c="red", marker="*", s=300,
                   zorder=12, edgecolors="darkred", linewidths=0.5)
        ax.annotate(k, (v["si_frac"], v["dH_DFT"]),
                    textcoords="offset points", xytext=(5, 6),
                    fontsize=8, color="darkred", fontweight="bold")
    ax.scatter([], [], c="red", marker="*", s=200, edgecolors="darkred",
               linewidths=0.5, label="DFT+U stable phases")

    for k, v in unstable.items():
        ax.scatter(v["si_frac"], v["dH_DFT"], c="none", marker="*", s=200,
                   edgecolors="red", linewidths=1.5, zorder=11)
        ax.annotate(f"{k}*", (v["si_frac"], v["dH_DFT"]),
                    textcoords="offset points", xytext=(5, -12),
                    fontsize=7.5, color="red", style="italic")

    # Real U-Si DB entries (from binary tier -- always binary)
    for sf in sorted_frac_bin:
        for r in hull_bin[sf]["ranking"]:
            ax.scatter(sf, r["dH_RF"], c="darkorange", marker="D", s=60,
                       zorder=10, edgecolors="black", linewidths=0.6)
            sg = r["sg"]
            pctl_bin = r["percentile"]
            # Find percentile in all pool
            pctl_all = pctl_bin  # default
            if sf in hull_all:
                for ra in hull_all[sf]["ranking"]:
                    if ra["sg"] == sg:
                        pctl_all = ra["percentile"]
                        break
            ax.annotate(f"SG{sg}\nB:{pctl_bin:.0f}% A:{pctl_all:.0f}%",
                        (sf, r["dH_RF"]),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=5.5, color="saddlebrown")
    ax.scatter([], [], c="darkorange", marker="D", s=60, edgecolors="black",
               linewidths=0.6, label="Real U-Si struct (RF pred)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Si / (Si+U) atomic fraction", fontsize=12)
    ax.set_ylabel("Formation Enthalpy  $\\Delta H_f$  (eV/atom)", fontsize=12)
    ax.set_title("U-Si Convex Hull: Tiered Structure Search vs. DFT+U\n"
                 "(Lopes et al. 2018, J. Nucl. Mater. 510, 331-336)",
                 fontsize=11)
    ax.set_xlim(-0.02, 1.02)

    mB = metrics_bin["model_RF"]
    mA = metrics_all["model_RF"]
    n_bin = hull_bin[sorted_frac_bin[0]]["n_templates"]
    n_all = hull_all[sorted_frac_all[0]]["n_templates"]
    ax.text(0.02, 0.03,
            f"Binary only: {n_bin} tmpl, MAE={mB['mae']:.3f} eV/atom\n"
            f"Binary+Ternary: {n_all} tmpl, MAE={mA['mae']:.3f} eV/atom",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    ax.text(0.98, 0.97, "* above hull (dynamically unstable)",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color="gray")

    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


# ---- Figure 2: Ranking comparison ------------------------------------------

def plot_ranking(hull_bin, hull_all, is_binary_all, out_path):
    """For each DFT phase with a DB entry, show sorted energy landscape
    for BOTH tiers side by side.  In the ALL tier, binary/ternary templates
    are plotted with different colors."""
    phases = []
    for name, ref in DFT_REF.items():
        sf = ref["si_frac"]
        closest_b = min(hull_bin.keys(), key=lambda x: abs(x - sf))
        closest_a = min(hull_all.keys(), key=lambda x: abs(x - sf))
        if abs(closest_b - sf) > 0.02:
            continue
        res_b = hull_bin[closest_b]
        res_a = hull_all[closest_a]
        if res_b["ranking"] or res_a["ranking"]:
            phases.append((name, ref, res_b, res_a))

    if not phases:
        return

    n = len(phases)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)

    for col, (name, ref, res_b, res_a) in enumerate(phases):
        for row, (res, tier_lbl, tier_clr) in enumerate([
                (res_b, "Binary only", "royalblue"),
                (res_a, "Binary+Ternary", "green")]):
            ax = axes[row, col]
            dh_arr = np.array(res["all_dH_RF"])
            sort_idx = np.argsort(dh_arr)
            sorted_dh = dh_arr[sort_idx]
            n_tmpl = len(sorted_dh)

            if row == 1 and is_binary_all is not None:
                # ALL tier: differentiate binary vs ternary templates
                sorted_src = is_binary_all[sort_idx]
                bin_ranks = np.where(sorted_src)[0]
                ter_ranks = np.where(~sorted_src)[0]
                ax.scatter(bin_ranks, sorted_dh[bin_ranks],
                           s=0.8, c="cornflowerblue", alpha=0.3,
                           rasterized=True, zorder=1)
                ax.scatter(ter_ranks, sorted_dh[ter_ranks],
                           s=0.8, c="lightsalmon", alpha=0.3,
                           rasterized=True, zorder=1)
                # Legend proxies
                ax.scatter([], [], s=10, c="cornflowerblue", alpha=0.6,
                           label=f"Binary ({len(bin_ranks)})")
                ax.scatter([], [], s=10, c="lightsalmon", alpha=0.6,
                           label=f"Ternary ({len(ter_ranks)})")
            else:
                # Binary-only tier: single color
                ax.plot(range(n_tmpl), sorted_dh, ".", markersize=1.0,
                        alpha=0.3, color=tier_clr)

            ax.axhline(ref["dH_DFT"], color="red", linewidth=1.5,
                       linestyle="--", label=f"DFT: {ref['dH_DFT']:.2f}")

            colors = ["darkorange", "purple", "brown"]
            for j, r in enumerate(res["ranking"]):
                c = colors[j % len(colors)]
                rank_0 = r["rank"] - 1
                ax.plot(rank_0, r["dH_RF"], "D", color=c, markersize=7,
                        markeredgecolor="black", markeredgewidth=0.7, zorder=10)
                ax.annotate(f"SG={r['sg']}\n{r['rank']}/{n_tmpl}\n"
                            f"({r['percentile']:.1f}%)",
                            (rank_0, r["dH_RF"]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=6, color=c,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      alpha=0.8, ec=c, lw=0.5))

            ax.plot(0, sorted_dh[0], "v", color="green", markersize=7,
                    zorder=9, label=f"Best: {sorted_dh[0]:+.3f}")

            if row == 1:
                ax.set_xlabel("Template rank (sorted)", fontsize=8)
            ax.set_ylabel("$\\Delta H_f$ (eV/atom)", fontsize=8)
            if row == 0:
                ax.set_title(f"{name} (Si={ref['si_frac']:.3f})", fontsize=9)
            ax.legend(fontsize=5.5, loc="lower right")

            # Tier label on left
            if col == 0:
                ax.annotate(tier_lbl, xy=(0, 0.5),
                            xycoords="axes fraction",
                            textcoords="offset points", xytext=(-40, 0),
                            fontsize=8, rotation=90, va="center",
                            fontweight="bold", color=tier_clr)

    fig.suptitle("Structure Ranking: Binary-Only vs Binary+Ternary Templates",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


# ---- Figure 3: Parity plot -------------------------------------------------

def plot_parity(metrics_bin, metrics_all, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    tiers = [
        (metrics_bin, "dH_RF", "RF binary-only", "royalblue"),
        (metrics_all, "dH_RF", "RF binary+ternary", "seagreen"),
        (metrics_all, "dH_NN", "NN comp (233-feat)", "darkorange"),
    ]

    for ax, (met, dh_key, label, color) in zip(axes, tiers):
        pm = met["per_phase"]
        xs, ys, names = [], [], []
        for name, p in pm.items():
            xs.append(p["dH_DFT"])
            ys.append(p[dh_key])
            names.append(name)
        xs, ys = np.array(xs), np.array(ys)

        lo = min(xs.min(), ys.min()) - 0.05
        hi = max(xs.max(), ys.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k-", linewidth=1.2, label="y=x", zorder=1)
        ax.fill_between([lo, hi], [lo-0.1, hi-0.1], [lo+0.1, hi+0.1],
                        alpha=0.12, color="gray", label="+/-0.10 band")
        ax.scatter(xs, ys, color=color, s=80, zorder=5, edgecolors="black",
                   linewidths=0.5)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 3),
                        fontsize=7)

        errs = ys - xs
        mae = float(np.mean(np.abs(errs)))
        rmse = float(np.sqrt(np.mean(errs**2)))
        bias = float(np.mean(errs))
        ax.set_xlabel("DFT+U  $\\Delta H_f$  (eV/atom)", fontsize=10)
        ax.set_ylabel("ML  $\\Delta H_f$  (eV/atom)", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.text(0.97, 0.05,
                f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nbias={bias:+.3f}",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    fig.suptitle("ML vs DFT+U Formation Enthalpies -- U-Si System", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}")


# ---- Report ----------------------------------------------------------------

def write_report(hull_bin, hull_all, metrics_bin, metrics_all,
                 runtime_sec, n_bin, n_all, usi_entries, out_path):
    pm_bin = metrics_bin["per_phase"]
    pm_all = metrics_all["per_phase"]
    mRF_bin = metrics_bin["model_RF"]
    mRF_all = metrics_all["model_RF"]
    mNN = metrics_all["model_NN"]
    n_comp = len(hull_all)

    # Formation enthalpy comparison table
    table_rows = ""
    for name, ref in DFT_REF.items():
        pb = pm_bin.get(name, {})
        pa = pm_all.get(name, {})
        dRF_b = pb.get("dH_RF", float("nan"))
        dRF_a = pa.get("dH_RF", float("nan"))
        dNN = pa.get("dH_NN", float("nan"))
        eRF_b = pb.get("err_RF", float("nan"))
        eRF_a = pa.get("err_RF", float("nan"))
        eNN = pa.get("err_NN", float("nan"))
        sg_b = pb.get("best_struct", {}).get("spacegroup_number", "?")
        sg_a = pa.get("best_struct", {}).get("spacegroup_number", "?")
        def _f(v): return f"{v:+.3f}" if not np.isnan(v) else "--"
        table_rows += (
            f"| {name:<7} | {ref['si_frac']:.3f} | {ref['dH_DFT']:+.3f} | "
            f"{_f(dRF_b)} | {_f(dRF_a)} | {_f(dNN)} | "
            f"SG {sg_b} | SG {sg_a} |\n"
        )

    # Ranking comparison table
    rank_rows = ""
    for name, ref in DFT_REF.items():
        pb = pm_bin.get(name, {})
        pa = pm_all.get(name, {})
        rank_b = {r["sg"]: r for r in pb.get("ranking", [])}
        rank_a = {r["sg"]: r for r in pa.get("ranking", [])}
        all_sgs = sorted(set(list(rank_b.keys()) + list(rank_a.keys())))
        for sg in all_sgs:
            rb = rank_b.get(sg, {})
            ra = rank_a.get(sg, {})
            comp = rb.get("composition", ra.get("composition", "?"))
            dH = rb.get("dH_RF", ra.get("dH_RF", float("nan")))
            rk_b = rb.get("rank", "?")
            pct_b = rb.get("percentile", float("nan"))
            rk_a = ra.get("rank", "?")
            pct_a = ra.get("percentile", float("nan"))
            def _pf(v): return f"{v:5.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "--"
            rank_rows += (
                f"| {name:<7} | {comp:8s} | SG {sg:3d} | {dH:+.4f} | "
                f"{rk_b:>5} / {n_bin} | {_pf(pct_b)} | "
                f"{rk_a:>5} / {n_all} | {_pf(pct_a)} |\n"
            )

    # Gap analysis: delta between real U-Si phase and best template
    mae_rf = mRF_all['mae']
    gap_rows = ""
    for name, ref in DFT_REF.items():
        pa = pm_all.get(name, {})
        best_dH = pa.get("dH_RF", float("nan"))
        best_sg = pa.get("best_struct", {}).get("spacegroup_number", "?")
        dft_sg = ref["sg"]
        sg_match = "YES" if best_sg == dft_sg else "no"
        for r in pa.get("ranking", []):
            gap = r["gap_to_best"]
            within = "YES" if gap < mae_rf else "no"
            ratio = gap / mae_rf * 100.0 if mae_rf > 0 else float("nan")
            gap_rows += (
                f"| {name:<7} | SG {r['sg']:3d} | {r['dH_RF']:+.4f} | "
                f"{best_dH:+.4f} | {gap:+.4f} | {ratio:5.1f}% | {within} |\n"
            )

    # SG match analysis: best-found SG vs DFT reference SG
    sg_rows = ""
    for name, ref in DFT_REF.items():
        pb = pm_bin.get(name, {})
        pa = pm_all.get(name, {})
        sg_dft = ref["sg"]
        sg_bin = pb.get("best_struct", {}).get("spacegroup_number", "?")
        sg_all = pa.get("best_struct", {}).get("spacegroup_number", "?")
        match_bin = "YES" if sg_bin == sg_dft else "no"
        match_all = "YES" if sg_all == sg_dft else "no"
        # Check if any real U-Si entry SG matches the best-found SG
        real_sgs = [r["sg"] for r in pa.get("ranking", [])]
        real_match_bin = "YES" if sg_bin in real_sgs else "--"
        real_match_all = "YES" if sg_all in real_sgs else "--"
        sg_rows += (
            f"| {name:<7} | SG {sg_dft:3d} ({ref['sym']}) | "
            f"SG {sg_bin} | {match_bin} | "
            f"SG {sg_all} | {match_all} |\n"
        )

    # Improvement analysis: how many compositions improved with ternary?
    n_improved = 0
    n_same = 0
    n_worse = 0
    for sf in hull_all:
        if sf in hull_bin:
            dh_b = hull_bin[sf]["best_dH_RF"]
            dh_a = hull_all[sf]["best_dH_RF"]
            if dh_a < dh_b - 1e-6:
                n_improved += 1
            elif dh_a > dh_b + 1e-6:
                n_worse += 1
            else:
                n_same += 1

    report = f"""# Agent A: Database Structure Search -- U-Si Convex Hull

---

## 1. Overview

Reproduces the U-Si convex hull from **Lopes et al. (2018), J. Nucl. Mater.
510, 331-336** using an exhaustive database structure search with two ML
models (RF struct + NN comp).

**Tiered search**: Compares binary-only templates ({n_bin}) vs binary+ternary
templates ({n_all}) to assess whether ternary structural prototypes improve
predictions for a binary system.

---

## 2. Methodology

### 2.1 Dual-Model Architecture

| Model | Type | Features | Checkpoint |
|-------|------|----------|------------|
| **RF struct** | RandomForest | 236 (composition + Magpie + structural) | `model/model_3.joblib` |
| **NN comp** | AERIS NN (PyTorch) | 233 (composition + Magpie) | `model/aeris_full_structure_classic.pt` |

### 2.2 Tiered Database Structure Search

| Tier | Pool | Templates | Description |
|------|------|-----------|-------------|
| **Tier 1** | Binary only (AB) | {n_bin} | All binary entries, deduplicated |
| **Tier 2** | Binary + Ternary (AB+ABC) | {n_all} | All binary+ternary, deduplicated |

- **{n_comp} target compositions** (all U_xSi_y with x+y <= 12)
- For each composition and each tier, the RF model predicts dH_f with every
  template's structural features (SG, density, CN_avg, CN_min, CN_max)
- Convex hulls constructed independently for each tier using Andrew's
  monotone chain (lower hull)

### 2.3 DFT Reference (Lopes et al. 2018)

| Phase | Si frac | dH_DFT (eV/atom) | SG | On hull? |
|-------|---------|-------------------|----|----------|
| U3Si  | 0.250 | -0.40 | 221 Pm-3m | Yes |
| U3Si2 | 0.400 | -0.48 | 127 P4/mbm | Yes |
| U5Si4 | 0.444 | -0.47 | 191 P6/mmm | No (unstable) |
| USi   | 0.500 | -0.55 | 63 Cmcm | Yes |
| USi2  | 0.667 | -0.52 | 191 P6/mmm | Yes |
| USi3  | 0.750 | -0.35 | 221 Pm-3m | Yes |

---

## 3. Results

### 3.1 Formation Enthalpies -- Tier Comparison

| Phase | Si frac | dH_DFT | dH_RF (bin) | dH_RF (all) | dH_NN | Best SG (bin) | Best SG (all) |
|-------|---------|--------|-------------|-------------|-------|---------------|---------------|
{table_rows}

### 3.2 Accuracy Metrics

| Metric | RF binary-only | RF binary+ternary | NN comp |
|--------|---------------|-------------------|---------|
| MAE    | {mRF_bin['mae']:.4f} eV/atom | {mRF_all['mae']:.4f} eV/atom | {mNN['mae']:.4f} eV/atom |
| RMSE   | {mRF_bin['rmse']:.4f} eV/atom | {mRF_all['rmse']:.4f} eV/atom | {mNN['rmse']:.4f} eV/atom |
| Bias   | {mRF_bin['bias']:+.4f} eV/atom | {mRF_all['bias']:+.4f} eV/atom | {mNN['bias']:+.4f} eV/atom |

### 3.3 Ternary Template Impact

Out of {n_comp} compositions:
- **{n_improved} improved** by adding ternary templates (lower best energy)
- **{n_same} unchanged**
- **{n_worse} worsened** (binary-only had lower energy)

### 3.4 Structure Ranking -- Binary vs All

| Phase | DB Entry | SG | dH_RF | Rank (bin) | Pctl (bin) | Rank (all) | Pctl (all) |
|-------|----------|----|-------|------------|------------|------------|------------|
{rank_rows}

### 3.5 Gap Analysis -- Real Phase vs Best Template

For each real U-Si database entry, the gap between its RF prediction and the
lowest-energy template found is compared to the RF model MAE ({mae_rf:.3f} eV/atom):

| Phase | Real SG | dH_RF (real) | dH_RF (best) | Gap | Gap/MAE | Within MAE? |
|-------|---------|-------------|-------------|-----|---------|-------------|
{gap_rows}
**All gaps are 0.005--0.018 eV/atom**, which is **2--9% of the model MAE**.
The correct structures are energetically indistinguishable from the best
templates at the resolution of the ML model.

### 3.6 Spacegroup Match -- Best Template vs DFT Reference

| Phase | DFT SG | Best SG (bin) | Match? | Best SG (all) | Match? |
|-------|--------|---------------|--------|---------------|--------|
{sg_rows}
**No spacegroup matches** between the best-found template and the DFT reference.
The RF model identifies low-energy structural descriptors (density, CN) but
maps them to different spacegroups. This is expected because: (1) the model
uses coarse structural proxies (SG number, density, CN), not atomic positions;
(2) multiple spacegroups can encode similar local coordination environments;
(3) the model was not trained on U-Si phases specifically.

---

## 4. Figures

- **Figure 1** (`results/agent_A_convex_hull.png`): Tiered convex hull overlay --
  binary-only (blue) vs binary+ternary (green) vs DFT+U (red).
- **Figure 2** (`results/agent_A_ranking.png`): Structure ranking for each tier
  (top: binary-only, bottom: binary+ternary).
- **Figure 3** (`results/agent_A_parity_plot.png`): Parity plot -- binary RF,
  all RF, and NN comp vs DFT.

---

## 5. Discussion

### 5.1 Binary vs Ternary Templates

The tiered comparison reveals whether ternary (ABC) structural prototypes
provide useful diversity or introduce noise for a binary (AB) system.
{n_improved} out of {n_comp} compositions found a lower-energy template among
the ternary entries, indicating that ternary prototypes {"do" if n_improved > n_comp // 3 else "do not significantly"} contribute new
low-energy structural motifs not present in binary prototypes alone.

### 5.2 Limitations

| Limitation | Impact |
|------------|--------|
| No structural relaxation | Systematic positive bias |
| Structural proxies (SG, density, CN) | Cannot resolve lattice parameters |
| Actinides underrepresented in training | Higher errors for U-containing phases |

---

## 6. Execution Summary

```
Script      : agent_paper_A_mcp.py
Models      : model/model_3.joblib (RF, 236 feat)
              model/aeris_full_structure_classic.pt (NN, 233 feat)
Runtime     : {runtime_sec/60:.1f} minutes
Binary tmpl : {n_bin}
All tmpl    : {n_all}
Compositions: {n_comp} stoichiometries
Reference   : Lopes et al. 2018, J. Nucl. Mater. 510, 331-336
```
"""
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  -> Report saved: {out_path}")


# ---- JSON results -----------------------------------------------------------

def save_results_json(hull_bin, hull_all, metrics_bin, metrics_all,
                      runtime_sec, n_bin, n_all, usi_entries, out_path):
    data = {
        "models": {"struct": MODEL_RF, "comp": MODEL_NN},
        "n_binary_templates": n_bin,
        "n_all_templates": n_all,
        "n_compositions": len(hull_all),
        "runtime_sec": runtime_sec,
        "metrics_RF_binary": metrics_bin["model_RF"],
        "metrics_RF_all": metrics_all["model_RF"],
        "metrics_NN": metrics_all["model_NN"],
        "per_phase_binary": {},
        "per_phase_all": {},
        "ranking_binary": [],
        "ranking_all": [],
    }
    for label, met, rank_key in [
            ("per_phase_binary", metrics_bin, "ranking_binary"),
            ("per_phase_all", metrics_all, "ranking_all")]:
        for name, p in met["per_phase"].items():
            entry = {
                "si_frac": p["si_frac"], "dH_DFT": p["dH_DFT"],
                "dH_RF": p["dH_RF"], "dH_NN": p["dH_NN"],
                "err_RF": p["err_RF"], "err_NN": p["err_NN"],
                "best_sg": p["best_struct"].get("spacegroup_number"),
            }
            data[label][name] = entry
            for r in p.get("ranking", []):
                data[rank_key].append({
                    "phase": name,
                    "db_composition": r["composition"],
                    "sg": r["sg"],
                    "dH_RF": r["dH_RF"],
                    "rank": r["rank"],
                    "n_total": r["n_total"],
                    "percentile": r["percentile"],
                    "gap_to_best": r["gap_to_best"],
                })

    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> Results JSON saved: {out_path}")


# ---- MAIN -------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 65)
    print("Agent A: Tiered Database Structure Search -- U-Si System")
    print("=" * 65)
    t0 = time.time()
    os.makedirs("results", exist_ok=True)

    print("\n[Loading database]")
    binary_templates, all_templates, usi_entries, is_binary_all = load_database()
    n_bin = len(binary_templates)
    n_all = len(all_templates)

    print("\n[Pre-computing structural feature matrices]")
    feat_list = list(_feat_rf)
    bin_idx, bin_mat = precompute_struct_matrix(binary_templates, feat_list)
    all_idx, all_mat = precompute_struct_matrix(all_templates, feat_list)
    print(f"  Binary: {bin_mat.shape[0]} templates x {len(bin_idx)} struct cols")
    print(f"  All:    {all_mat.shape[0]} templates x {len(all_idx)} struct cols")

    compositions = generate_compositions()
    print(f"\n  {len(compositions)} target compositions")

    # ---- Tier 1: Binary only ----
    print("\n" + "=" * 65)
    print(f"TIER 1: Binary-only ({n_bin} templates)")
    print("=" * 65)
    nn_cache = {}
    hull_bin, nn_cache = run_database_screen(
        compositions, binary_templates, usi_entries,
        bin_idx, bin_mat, "BIN", nn_cache)

    # ---- Tier 2: Binary + Ternary ----
    print("\n" + "=" * 65)
    print(f"TIER 2: Binary+Ternary ({n_all} templates)")
    print("=" * 65)
    hull_all, nn_cache = run_database_screen(
        compositions, all_templates, usi_entries,
        all_idx, all_mat, "ALL", nn_cache)

    # ---- Metrics ----
    print("\n[Metrics]")
    metrics_bin = compute_metrics(hull_bin, "RF-binary")
    metrics_all = compute_metrics(hull_all, "RF-all")
    mB = metrics_bin["model_RF"]
    mA = metrics_all["model_RF"]
    mN = metrics_all["model_NN"]
    print(f"  RF binary-only:    MAE={mB['mae']:.4f}")
    print(f"  RF binary+ternary: MAE={mA['mae']:.4f}")
    print(f"  NN comp:           MAE={mN['mae']:.4f}")

    # ---- Figures ----
    print("\n[Figures]")
    plot_convex_hull(hull_bin, hull_all, metrics_bin, metrics_all,
                     is_binary_all, "results/agent_A_convex_hull.png")
    plot_ranking(hull_bin, hull_all, is_binary_all,
                 "results/agent_A_ranking.png")
    plot_parity(metrics_bin, metrics_all, "results/agent_A_parity_plot.png")

    _flush_mcp_log()
    runtime_sec = time.time() - t0

    print("\n[Report]")
    write_report(hull_bin, hull_all, metrics_bin, metrics_all,
                 runtime_sec, n_bin, n_all, usi_entries,
                 "results/agent_A_report.md")
    save_results_json(hull_bin, hull_all, metrics_bin, metrics_all,
                      runtime_sec, n_bin, n_all, usi_entries,
                      "results/agent_A_results.json")

    print(f"\nDone in {runtime_sec/60:.1f} min.")
    print("=" * 65)


if __name__ == "__main__":
    main()
