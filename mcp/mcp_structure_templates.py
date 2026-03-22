# mcp_structure_templates.py
"""
An MCP tool to search for template structure candidates that would fit 
a given composition based on entries in Dataset*.csv files.
"""

from typing import Any, Optional

from fastmcp import FastMCP

from aeris import read_datasets, find_structure_templates

mcp = FastMCP(name="FindStructureTemplates")


@mcp.tool
def template_candidates(composition: str, spacegroup: Optional[int] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """
    Searches for all compatible structures in Dataset*.csv files and returns all matching entries.

    Args:
      composition: The chemical composition to search for (e.g., "UO2")
      spacegroup: Optional filter entries by a given space group
      limit: Optional maximum number of entries to return. If None or <= 0, returns all matches.

    Returns:
      A list of dictionaries, each with materials_id and compositions.
      candidates = {
            'composition': row['composition'],
            'structure': row['structure'],
            'spacegroup_number': row['spacegroup_number'],
            'density_atomic': row['density_atomic'],
            'CN_max': row['CN_max'],
            'CN_min': row['CN_min'],
            'CN_avg': row['CN_avg'],
            "source_file": file_path
        }
      Returns an empty dictionary if no matches are found.
    """

    # identify all the entries in the datasets with the given composition
    df = read_datasets("Dataset*.csv")
    if spacegroup is not None:
        df = df[df.spacegroup_number == spacegroup]
    template_entries = find_structure_templates(df, composition)

    # Respect the optional limit if provided
    if limit is not None and limit > 0 and len(template_entries) >= limit:
        return template_entries[:limit]
    return template_entries


if __name__ == "__main__":
    mcp.run()

