# Agent A: Database Structure Search -- U-Si Convex Hull

---

## 1. Overview

Reproduces the U-Si convex hull from **Lopes et al. (2018), J. Nucl. Mater.
510, 331-336** using an exhaustive database structure search with two ML
models (RF struct + NN comp).

**Tiered search**: Compares binary-only templates (8742) vs binary+ternary
templates (31476) to assess whether ternary structural prototypes improve
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
| **Tier 1** | Binary only (AB) | 8742 | All binary entries, deduplicated |
| **Tier 2** | Binary + Ternary (AB+ABC) | 31476 | All binary+ternary, deduplicated |

- **45 target compositions** (all U_xSi_y with x+y <= 12)
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
| U3Si    | 0.250 | -0.400 | -0.186 | -0.186 | -0.137 | SG 140 | SG 140 |
| U3Si2   | 0.400 | -0.480 | -0.200 | -0.200 | -0.323 | SG 11 | SG 11 |
| U5Si4   | 0.444 | -0.470 | -0.269 | -0.269 | -0.364 | SG 15 | SG 15 |
| USi     | 0.500 | -0.550 | -0.318 | -0.318 | -0.403 | SG 10 | SG 10 |
| USi2    | 0.667 | -0.520 | -0.347 | -0.348 | -0.423 | SG 186 | SG 189 |
| USi3    | 0.750 | -0.350 | -0.219 | -0.219 | -0.374 | SG 2 | SG 63 |


### 3.2 Accuracy Metrics

| Metric | RF binary-only | RF binary+ternary | NN comp |
|--------|---------------|-------------------|---------|
| MAE    | 0.2054 eV/atom | 0.2053 eV/atom | 0.1321 eV/atom |
| RMSE   | 0.2106 eV/atom | 0.2105 eV/atom | 0.1507 eV/atom |
| Bias   | +0.2054 eV/atom | +0.2053 eV/atom | +0.1242 eV/atom |

### 3.3 Ternary Template Impact

Out of 45 compositions:
- **17 improved** by adding ternary templates (lower best energy)
- **28 unchanged**
- **0 worsened** (binary-only had lower energy)

### 3.4 Structure Ranking -- Binary vs All

| Phase | DB Entry | SG | dH_RF | Rank (bin) | Pctl (bin) | Rank (all) | Pctl (all) |
|-------|----------|----|-------|------------|------------|------------|------------|
| U3Si    | U3Si1    | SG 140 | -0.1786 |  1238 / 8742 |  14.2% |  4592 / 31476 |  14.6% |
| U3Si    | U3Si1    | SG 221 | -0.1684 |  3911 / 8742 |  44.7% | 13041 / 31476 |  41.4% |
| U3Si2   | U3Si2    | SG 127 | -0.1923 |   490 / 8742 |   5.6% |   924 / 31476 |   2.9% |
| USi     | U1Si1    | SG  62 | -0.3101 |   876 / 8742 |  10.0% |  1410 / 31476 |   4.5% |
| USi2    | U1Si2    | SG 141 | -0.3295 |  4985 / 8742 |  57.0% | 13896 / 31476 |  44.1% |
| USi2    | U1Si2    | SG 191 | -0.3430 |   748 / 8742 |   8.6% |  1764 / 31476 |   5.6% |
| USi3    | U1Si3    | SG 221 | -0.2086 |  4086 / 8742 |  46.7% | 11852 / 31476 |  37.7% |


### 3.5 Gap Analysis -- Real Phase vs Best Template

For each real U-Si database entry, the gap between its RF prediction and the
lowest-energy template found is compared to the RF model MAE (0.205 eV/atom):

| Phase | Real SG | dH_RF (real) | dH_RF (best) | Gap | Gap/MAE | Within MAE? |
|-------|---------|-------------|-------------|-----|---------|-------------|
| U3Si    | SG 221 | -0.1684 | -0.1855 | +0.0172 |   8.4% | YES |
| U3Si    | SG 140 | -0.1786 | -0.1855 | +0.0069 |   3.4% | YES |
| U3Si2   | SG 127 | -0.1923 | -0.1998 | +0.0075 |   3.7% | YES |
| USi     | SG  62 | -0.3101 | -0.3176 | +0.0075 |   3.7% | YES |
| USi2    | SG 141 | -0.3295 | -0.3478 | +0.0183 |   8.9% | YES |
| USi2    | SG 191 | -0.3430 | -0.3478 | +0.0047 |   2.3% | YES |
| USi3    | SG 221 | -0.2086 | -0.2191 | +0.0105 |   5.1% | YES |

**All gaps are 0.005--0.018 eV/atom**, which is **2--9% of the model MAE**.
The correct structures are energetically indistinguishable from the best
templates at the resolution of the ML model.

### 3.6 Spacegroup Match -- Best Template vs DFT Reference

| Phase | DFT SG | Best SG (bin) | Match? | Best SG (all) | Match? |
|-------|--------|---------------|--------|---------------|--------|
| U3Si    | SG 221 (Pm-3m) | SG 140 | no | SG 140 | no |
| U3Si2   | SG 127 (P4/mbm) | SG 11 | no | SG 11 | no |
| U5Si4   | SG 191 (P6/mmm) | SG 15 | no | SG 15 | no |
| USi     | SG  63 (Cmcm) | SG 10 | no | SG 10 | no |
| USi2    | SG 191 (P6/mmm) | SG 186 | no | SG 189 | no |
| USi3    | SG 221 (Pm-3m) | SG 2 | no | SG 63 | no |

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
17 out of 45 compositions found a lower-energy template among
the ternary entries, indicating that ternary prototypes do contribute new
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
Runtime     : 1.7 minutes
Binary tmpl : 8742
All tmpl    : 31476
Compositions: 45 stoichiometries
Reference   : Lopes et al. 2018, J. Nucl. Mater. 510, 331-336
```
