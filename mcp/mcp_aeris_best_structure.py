# mcp_aeris_best_structure.py
"""
An MCP tool to search for best structures for a given composition
using the surrogate model saved at the given model path.
"""

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from aeris import find_energy_forall_structures

mcp = FastMCP(name="PredictCompositionEnergy")


@mcp.tool
def total_energy(composition: str, model_path: str, limit_energy: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Computes the total energy for a composition and all compatible structures from a dataset
    using the model provided.

    Args:
      composition: a formula of the chemical composition like "Fe2O3"
      model_path: the path to the model used to predict energy
      limit_energy: optional threshold for acceptable energies for the output composition/structure pair

    Returns:
      A list of dictionary with:
        'per_atom_eV_per_atom': per_atom,
        'total_eV': total_e,
        'parsed_composition': parsed,
        'n_atoms': n_atoms,
        'source': 'predicted'
        'device': device
    """

    results = find_energy_forall_structures('Dataset*.csv', model_path, composition)
    if limit_energy is not None and limit_energy > 0:
        return [i for i in results if i['per_atom_eV_per_atom']>limit_energy]
    return results

if __name__ == "__main__":
    mcp.run()
