---
name: reaction-energy
description: Calculate interfacial reaction enthalpies using dual ML models (RF + NN). Evaluates thermodynamic compatibility via Hess's law. Invoke with /reaction-energy [reaction equation].
argument-hint: "[reaction, e.g. UN + 2V = V2N + U] or [system, e.g. UN vs V,Nb,Ta]"
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Reaction Energy Calculator

Compute reaction enthalpies for material interfaces using dual ML formation-energy models. Determines whether an interface is thermodynamically stable (endothermic) or will react (exothermic).

## What This Skill Does

1. Parses reaction equations and identifies all phases
2. Looks up structural descriptors from the database or CIF files
3. Predicts formation enthalpies with both RF (struct) and NN (comp) models
4. Computes reaction enthalpy via Hess's law (pure elements = 0 by definition)
5. Reports sign agreement and compatibility ordering

## How to Run

The reference implementation covers 26 reactions across 6 metals:

```bash
# From project root
python validation/paper_b/agent_paper_B.py
```

**For new reactions**: read `validation/paper_b/agent_paper_B.py` for the pattern. Key logic:
- `dH_rxn = (sum products - sum reactants) / N_total_atoms`
- Positive = stable interface, negative = will react
- Pure element formation enthalpy = 0 (override model predictions)

## Interpretation

| dH_rxn | Meaning |
|--------|---------|
| **> 0** (endothermic) | Thermodynamically **stable** (compatible) |
| **< 0** (exothermic) | Thermodynamically **unstable** (will react) |
| ~ 0 | Marginal -- depends on kinetics |

For multi-reaction systems, the **minimum** dH_rxn determines overall compatibility.

## Project Resources

| Resource | Path |
|----------|------|
| Reference script | `validation/paper_b/agent_paper_B.py` |
| Core library | `aeris.py` |
| RF model (236 feat) | `model/model_3.joblib` |
| NN model (230 feat) | `model/aeris_comp_only.pt` |
| NN struct model (236 feat) | `model/aeris_full_struct.pt` |
| Structure database | `data/Dataset_feature+CN.csv` |
| CIF files | `data/cif/` |

## Model Recommendations

| Use case | Recommended model |
|----------|-------------------|
| Go/no-go compatibility | NN comp (reaction MAE 0.109 eV/atom) |
| Sign-critical decisions | RF struct (96% sign agreement) |
| Defect ranking | RF struct (52% top-1 accuracy) |
| When models disagree on sign | Flag as uncertain, report magnitudes |

## Arguments

Parse `$ARGUMENTS` as reaction equation(s) or a system to evaluate.
If no reaction given, ask what system to evaluate.

## Validated Results (UN vs V, Nb, Ta, Cr, Mo, W)

- RF sign agreement: 25/26 reactions (96%)
- NN reaction MAE: 0.109 eV/atom
- Mo and W correctly identified as most stable cladding
- Reference: Hua et al. 2022, J. Nucl. Mater. 560, 153462
- Full report: `validation/paper_b/agent_B_report.md`
