# AERIS Validation Project

ML-based formation enthalpy prediction for nuclear materials using dual-model architecture (RF struct-aware + NN composition-only).

## Project Structure

```
AERIS_validation/
├── CLAUDE.md              # This file
├── AGENTS.md              # Agent definitions
├── README.md              # Project overview
├── aeris.py               # Core library (models, featurizers, parsers)
├── .claude/skills/        # Claude Code skills
│   ├── structure-search.md   # Binary convex hull predictor
│   ├── reaction-energy.md    # Reaction enthalpy calculator
│   └── point-defect.md       # Point-defect formation energies
├── mcp/                   # MCP tool servers (FastMCP)
│   ├── mcp_aeris_predict.py         # Predict energy for composition+structure
│   ├── mcp_aeris_best_structure.py  # Search best structures for composition
│   ├── mcp_structure_search.py      # Search dataset for composition matches
│   └── mcp_structure_templates.py   # Find structural templates
├── model/                 # Trained ML models
│   ├── model_3.joblib                    # RF (236 features, struct-aware)
│   ├── aeris_comp_only.pt                # NN (230 features, comp-only)
│   └── aeris_full_struct.pt              # NN (236 features, struct-aware)
├── data/                  # Datasets and CIF files
│   ├── Dataset_feature+CN.csv   # 48,490 entries
│   └── cif/                     # Metal BCC + nitride CIF files
├── reference/             # Source papers (PDF)
│   ├── Lopes et al. 2018, J. Nucl. Mater. 510, 331-336.pdf  # Paper A ref
│   └── Hua et al. 2022, J. Nucl. Mater. 560, 153462.pdf       # Paper B ref
├── training/              # Model training scripts
│   └── train_nn.py        # Train NN comp-only and full-struct models
└── validation/            # Paper reproduction results
    ├── paper_a/           # U-Si convex hull (Lopes et al. 2018)
    └── paper_b/           # UN/metal compatibility (Hua et al. 2022)
```

## Models

| Model | File | Features | Best For |
|-------|------|----------|----------|
| RF struct | `model/model_3.joblib` | 236 (comp + Magpie + structural) | Structure-aware predictions, defect ranking, sign agreement |
| NN comp | `model/aeris_comp_only.pt` | 230 (comp + Magpie) | Formation/reaction enthalpy screening (val MAE 0.124) |
| NN struct | `model/aeris_full_struct.pt` | 236 (same as RF) | NN with structural features (val MAE 0.170) |

## Key Library Functions (aeris.py)

- `load_structure_model(path, device)` - Load RF or NN model, returns (model, feature_names, scaler)
- `parse_formula(formula)` - Parse "U3Si2" -> {"U": 3, "Si": 2}
- `compute_magpie_df(compositions)` - Compute Magpie descriptors for a Series of compositions
- `predict_energy(model, features)` - Single prediction with structure
- `find_structure_in_datasets(df, composition)` - Search dataset for exact matches
- `find_structure_templates(df, composition)` - Find compatible structural templates

## Skills

Three local skills are available:

- `/structure-search [A-B]` - Exhaustive binary convex hull search (~31k templates, dual-model)
- `/reaction-energy [reaction equation]` - Interfacial reaction enthalpy via Hess's law
- `/point-defect [host] [defect] [equilibrium]` - Point-defect formation energies via 3-phase equilibrium

## MCP Servers

Run any MCP tool with: `python mcp/<tool>.py`

All MCP tools use FastMCP and import from `aeris.py` at the project root.

## Dependencies

- Python 3.10+
- PyTorch, scikit-learn, numpy, pandas
- pymatgen, matminer (Magpie featurizers)
- fastmcp (MCP tools)
- matplotlib (plotting)

## Running Scripts

Always run from the project root:
```bash
cd AERIS_validation
python validation/paper_a/agent_paper_A_mcp.py
python validation/paper_b/agent_paper_B.py
```

## Validated Results

- **Paper A** (U-Si): RF MAE 0.205, NN MAE 0.132 eV/atom. All real phases rank top 3-15%.
- **Paper B** (UN/metals): RF sign agreement 96% (25/26 reactions). NN reaction MAE 0.109 eV/atom.

## References

Source papers used for validation (PDFs in `reference/`):

- **Paper A**: Lopes et al. 2018, *J. Nucl. Mater.* 510, 331-336 -- U-Si convex hull via DFT+U
- **Paper B**: Hua et al. 2022, *J. Nucl. Mater.* 560, 153462 -- UN/metal thermodynamic compatibility
