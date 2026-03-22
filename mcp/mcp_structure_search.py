# mcp_structure_search.py
"""
An MCP tool to search for structure and total energy of a composition in Dataset*.csv files.
"""

from typing import Any, Optional

import numpy as np
from fastmcp import FastMCP

from aeris import read_datasets, parse_formula, find_structure_in_datasets

mcp = FastMCP(name="StructureSearchInDataset")


@mcp.tool
def structure_and_energy(composition: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """
    Searches for a composition in Dataset*.csv files and returns all matching entries.

    Args:
      composition: The chemical composition to search for (e.g., "UO2").
      limit: Optional maximum number of entries to return. If None or <= 0, returns all matches.

    Returns:
      A list of dictionaries, each with energy and composition information.
      Returns an empty list if no matches are found.
    """

    # identify all the entries in the datasets with the given composition
    df = read_datasets("Dataset*.csv")
    entries = find_structure_in_datasets(df, composition)

    # identify each component and the atoms per component in the given composition
    parsed = parse_formula(composition)
    n_atoms = float(sum(parsed.values()))

    # create dictionary with all the results that have energy recorded
    candidates = [{
        'per_atom_eV_per_atom': i["per_atom_eV_per_atom"],
        'total_eV': i["total_eV"],
        'parsed_composition': parsed,
        'n_atoms': n_atoms,
        'source': "dataset",
    } for i in entries if not np.isnan(i["per_atom_eV_per_atom"])]

    # Respect the optional limit if provided
    if limit is not None and limit > 0 and len(candidates) >= limit:
        return candidates[:limit]
    return candidates

if __name__ == "__main__":
    mcp.run()
