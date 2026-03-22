# AERIS Agents

## Agent A: Structure Search Agent

**Purpose**: Exhaustive database structure search for binary material systems.

**Skill**: `.claude/skills/structure-search.md`

**What it does**:
- Screens ~31,000 structural templates from the crystallographic database
- Uses dual ML models (RF struct-aware + NN composition-only)
- Predicts the formation-enthalpy convex hull for any binary A-B system
- Supports tiered search: binary-only (8,742 templates) vs binary+ternary (31,476 templates)
- Ranks real database entries against all templates
- Produces convex hull plots and JSON results

**Validated on**: U-Si system (Lopes et al. 2018, J. Nucl. Mater. 510, 331-336)

**Reference script**: `validation/paper_a/agent_paper_A_mcp.py`

**Invoke**: `/structure-search U-Si` or ask "search structures for the Zr-O binary system"

---

## Agent B: Reaction Energy Agent

**Purpose**: Calculate interfacial reaction enthalpies between compounds using ML formation-energy models.

**Skill**: `.claude/skills/reaction-energy.md`

**What it does**:
- Evaluates thermodynamic compatibility between materials
- Computes reaction enthalpy via Hess's law from ML-predicted formation enthalpies
- Uses dual-model architecture for cross-validation (RF struct + NN comp)
- Supports point-defect formation energy calculations
- Reports sign agreement and metal compatibility ordering

**Validated on**: UN/metal compatibility (Hua et al. 2022, J. Nucl. Mater. 560, 153462)

**Reference script**: `validation/paper_b/agent_paper_B.py`

**Invoke**: `/reaction-energy UN + 2V = V2N + U` or ask "is UN compatible with vanadium cladding?"

---

## MCP Tools

Four MCP tool servers provide programmatic access to the AERIS models:

| Tool | Script | Description |
|------|--------|-------------|
| Predict Energy | `mcp/mcp_aeris_predict.py` | Predict formation energy for composition + structure |
| Best Structure | `mcp/mcp_aeris_best_structure.py` | Search best structures across all templates |
| Structure Search | `mcp/mcp_structure_search.py` | Find exact composition matches in dataset |
| Structure Templates | `mcp/mcp_structure_templates.py` | Find compatible structural templates |

Run any server: `python mcp/<tool>.py`
