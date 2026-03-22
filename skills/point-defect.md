---
name: point-defect
description: Calculate point-defect formation energies in compounds using 3-phase equilibrium chemical potentials and supercell composition proxies. Supports vacancies, interstitials, and substitutional defects.
argument-hint: "[host compound] [defect type] [equilibrium phases, e.g. UN-V-VN]"
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Point-Defect Formation Energy Calculator

Calculate point-defect formation energies in crystalline compounds using ML models with supercell composition proxies and 3-phase equilibrium chemical potentials.

## What This Skill Does

1. Defines a supercell of the host compound (e.g. U32N32 for UN)
2. Creates defect compositions by adding/removing atoms (e.g. U31N32 for U vacancy)
3. Solves 3-phase equilibrium for chemical potentials via linear algebra
4. Computes defect formation energy: `E_f = E_defect - E_perfect - sum(delta_n * mu)`
5. Ranks defect types by formation energy (lowest = most likely defect)

## Defect Types

| Defect | Composition change | Structural context (RF) |
|--------|-------------------|------------------------|
| Vacancy A | Remove 1 A atom | Host structure |
| Vacancy B | Remove 1 B atom | Host structure |
| Interstitial B | Add 1 B atom | Host structure |
| X on A (substitution) | Remove 1 A, add 1 X | Substituent metal structure |
| X on B (substitution) | Remove 1 B, add 1 X | Substituent metal structure |

## How to Calculate

The reference implementation is in `validation/paper_b/agent_paper_B.py` (Phase 4: defect calculations). Key functions:

- `solve_mu(E, phase1, phase2, phase3, elements)` -- Solve 3-phase equilibrium for chemical potentials
- `calc_defect_formation(E_def, E_perf, mu, deltas)` -- Compute E_f from defect/perfect energies and chemical potentials

### Chemical Potential Calculation

Given 3 coexisting phases (e.g. UN, V, VN), solve:

```
A * mu = b
where A[i,j] = number of element j in phase i
      b[i]   = total energy of phase i (= E_per_atom * n_atoms)
```

### Defect Formation Energy

```
E_f = E_defect - E_perfect - sum(delta_n_i * mu_i)
```

Where:
- `E_defect` = total energy of defect supercell (ML prediction * n_atoms)
- `E_perfect` = total energy of perfect supercell
- `delta_n_i` = change in number of atoms of element i (+1 for added, -1 for removed)
- `mu_i` = chemical potential of element i from 3-phase equilibrium

### Structural Context for RF Model

- **Vacancies and interstitials**: Use host compound structure (e.g. UN rocksalt SG 225)
- **Substitutional defects (X on site)**: Use substituent metal structure (e.g. BCC SG 229 from CIF)
- This captures the local structural environment introduced by the foreign atom

## Project Resources

| Resource | Path |
|----------|------|
| Reference script | `validation/paper_b/agent_paper_B.py` |
| Core library | `aeris.py` |
| RF model (236 feat) | `model/model_3.joblib` |
| NN model (230 feat) | `model/aeris_comp_only.pt` |
| Structure database | `data/Dataset_feature+CN.csv` |
| Extended database | `data/Dataset_feture+CN_exp.csv` |
| CIF files | `data/cif/` |

## Model Recommendations

| Use case | Model | Accuracy |
|----------|-------|----------|
| Defect ranking | RF struct | 52% top-1, Kendall tau +0.35 |
| Defect absolute values | RF struct | MAE 2.12 eV |
| Composition-only baseline | NN comp | 28% top-1, MAE 8.6 eV |

RF struct is strongly preferred for defect calculations due to structural context.

## Arguments

Parse `$ARGUMENTS` as:
- Host compound + defect type + equilibrium phases (e.g. `UN vacancy U in UN-V-VN`)
- Or a full system scan (e.g. `all defects in UN with V cladding`)
- If unclear, ask what host compound and equilibrium to evaluate

## Validated Results (UN system)

- 25 three-phase equilibria across 6 metals (V, Nb, Ta, Cr, Mo, W)
- RF struct top-1 accuracy: 13/25 (52%), mean Kendall tau: +0.352
- V/Nb systems: VacN correctly predicted as dominant defect (7/9)
- Cr systems: VacU correctly predicted as dominant (4/5)
- Reference: Hua et al. 2022, J. Nucl. Mater. 560, 153462
- Full report: `validation/paper_b/agent_B_report.md`
