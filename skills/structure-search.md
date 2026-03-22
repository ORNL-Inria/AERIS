---
name: structure-search
description: Binary convex hull prediction via exhaustive database structure search with dual ML models (RF + NN). Screens ~31k structural templates. Invoke with /structure-search [A-B system].
argument-hint: "[A-B system, e.g. U-Si, Zr-O] [optional: DFT reference values]"
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Structure Search -- Binary Convex Hull

Predict the formation-enthalpy convex hull for a binary A-B system by screening ~31,000 structural templates with dual ML models.

## What This Skill Does

1. Generates all stoichiometries A_xB_y with x+y <= 12 (typically 45 compositions)
2. Screens every composition against ~8,700 binary and ~31,500 binary+ternary structural templates
3. Predicts formation enthalpy with RF (struct-aware) and NN (comp-only) models
4. Builds the lower convex hull for each tier
5. Ranks real database entries against the template pool
6. Produces plots and a markdown report

## How to Run

The reference implementation is fully working. Adapt and run it:

```bash
# From project root
python validation/paper_a/agent_paper_A_mcp.py
```

**For a new A-B system**: read `validation/paper_a/agent_paper_A_mcp.py`, then create a copy under `results/` with these changes:
- Set `EL_A` and `EL_B` to the target elements
- Update or remove `DFT_REF` (set to `None` if no reference data)
- Adjust `MAX_ATOMS` if needed (default 12)

## Project Resources

| Resource | Path |
|----------|------|
| Reference script | `validation/paper_a/agent_paper_A_mcp.py` |
| Core library | `aeris.py` |
| RF model (236 feat) | `model/model_3.joblib` |
| NN model (230 feat) | `model/aeris_comp_only.pt` |
| NN struct model (236 feat) | `model/aeris_full_struct.pt` |
| Structure database | `data/Dataset_feature+CN.csv` (48,490 entries) |

## Key Functions (aeris.py)

- `load_structure_model(path, device)` -- loads RF (.joblib) or NN (.pt), returns (model, feature_names, scaler)
- `parse_formula(formula)` -- "U3Si2" -> {"U": 3, "Si": 2}
- `compute_magpie_df(compositions)` -- Magpie descriptors for a pd.Series

## Arguments

Parse `$ARGUMENTS` as an A-B system (e.g. `U-Si`, `Zr-O`, `Ti-Al`).
Optional DFT reference after the system: `U-Si with DFT: U3Si=-0.40, U3Si2=-0.48`.
If no system given, ask.

## Validated Results (U-Si)

- RF MAE: 0.205 eV/atom, NN MAE: 0.132 eV/atom
- All real U-Si phases rank in top 3-15% of template pool
- Gap to best template: 0.005-0.018 eV/atom (2-9% of model MAE)
- Reference: Lopes et al. 2018, J. Nucl. Mater. 510, 331-336
- Full report: `validation/paper_a/agent_A_report.md`
