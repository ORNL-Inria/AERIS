# mcp_aeris_predict.py
"""
An MCP tool to predict the energy of a composition with a given structure
using the surrogate model saved at the given model path.
"""

from typing import Any, Dict

from fastmcp import FastMCP

from aeris import parse_formula, predict_energy

mcp = FastMCP(name="PredictCompositionEnergy")


@mcp.tool
def total_energy(composition: str, model_path: str, features: Dict) -> Dict[str, Any]:
    """
    Computes the total energy for a composition and a structure using the model provided.

    Args:
      composition: a formula of the chemical composition like "Fe2O3"
      model_path: the path to the model used to predict energy
      features: optional structure info dictionary used to build structural features

    Returns:
      A dictionary with:
        'per_atom_eV_per_atom': per_atom,
        'total_eV': total_e,
        'parsed_composition': parsed,
        'n_atoms': n_atoms,
        'source': 'predicted'
        'device': device
    """

    out = predict_energy(composition, model_path, features)
    parsed = parse_formula(composition)
    out['parsed_composition'] = parsed
    out['n_atoms'] = float(sum(parsed.values()))
    out['source'] = "predicted"
    return out


if __name__ == "__main__":
    mcp.run()
