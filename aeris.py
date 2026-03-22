import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from mpi4py import MPI
    AERIS_HAS_MPI = True
except:
    AERIS_HAS_MPI = False
    print("MPI not found. Running serial version")

from pymatgen.core.composition import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf


__all__ = [
    "AerisFullStructure",
    "read_datasets",
    "parse_formula",
    "parse_structure_string",
    "load_structure_model",
    "build_features_in_ckpt_order",
    "predict_energy",
    "find_structure_in_datasets",
    "find_structure_templates",
    "find_energy_forall_structures",
]

def read_datasets(dataset_files: str) -> List[Any]:
    """Function to read a list of csv files.
    Input: partial name (string) of the dataset
    Output: pandas array with all the entries
    """
    dfs = []
    dataset_files = glob.glob(dataset_files)
    for file_path in dataset_files:
        dfs.append(pd.read_csv(file_path, low_memory=False))
    return pd.concat(dfs, ignore_index=True)

def parse_formula(s: str) -> Dict[str, float]:
    """Function to retrurn the formula of a given composition.
    Input: the composition string (e.g. UO2)
    Output: a dictionary {element: number of elements} (e.g. {U: 1, O: 2})
    """
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(s).strip())
    if not parts:
        raise ValueError(f"Could not parse formula: {s}")
    comp: Dict[str, float] = {}
    for el, num in parts:
        comp[el] = float(num) if num else 1.0
    return comp

def _apply_df_parse_formula_num(val):
    try:
        if pd.isna(val):
            return None
        parsed = parse_formula(str(val))
        return int(sum(parsed.values()))
    except Exception:
        return None

def _apply_df_parse_formula_str(val):
    try:
        if pd.isna(val):
            return None
        parsed = parse_formula(str(val))
        return ''.join(f"{k}{v}" for k, v in sorted(parsed.items()))
    except Exception:
        return None

def parse_structure_string(struct_str: str) -> Dict[str, float]:
    """Function to read a structure string.
    Input: structure string
    Output: dictionary with structure features
    """
    # minimal lattice extractor (compatible with training utils)
    result = {
        'lattice_a': np.nan,
        'lattice_b': np.nan,
        'lattice_c': np.nan,
        'lattice_alpha': np.nan,
        'lattice_beta': np.nan,
        'lattice_gamma': np.nan,
        'volume': np.nan,
        'density': np.nan,
        'nsites': np.nan,
        'spacegroup_number': np.nan,
    }
    if struct_str is None:
        return result
    s = str(struct_str)
    abc_pattern = r'abc\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    angles_pattern = r'angles\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    abc = re.search(abc_pattern, s)
    ang = re.search(angles_pattern, s)
    if abc:
        result['lattice_a'] = float(abc.group(1))
        result['lattice_b'] = float(abc.group(2))
        result['lattice_c'] = float(abc.group(3))
    if ang:
        result['lattice_alpha'] = float(ang.group(1))
        result['lattice_beta'] = float(ang.group(2))
        result['lattice_gamma'] = float(ang.group(3))
    # try volume
    vol_match = re.search(r'volume\s*[:=]\s*([\d.]+)', s)
    if vol_match:
        result['volume'] = float(vol_match.group(1))
    dens_match = re.search(r'density\s*[:=]\s*([\d.]+)', s)
    if dens_match:
        result['density'] = float(dens_match.group(1))
    sg_match = re.search(r'spacegroup(?:_number)?\s*[:=]\s*(\d+)', s)
    if sg_match:
        result['spacegroup_number'] = int(sg_match.group(1))
    nsites_match = re.search(r'nsites\s*[:=]\s*(\d+)', s)
    if nsites_match:
        result['nsites'] = int(nsites_match.group(1))
    return result


class AerisFullStructure(nn.Module):
    """The Aeris model architecture used for prediction
    """
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        first_layer = min(1024, max(512, input_dim * 2))
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first_layer), nn.ReLU(), nn.BatchNorm1d(first_layer),
            nn.Linear(first_layer, first_layer), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(first_layer, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

def load_structure_model(model_file: str, device: str = 'cpu'):
    """Function to load the model.
    Input: the path to the model checkpoint that contains a dictionary with
    the scalar, input dimensions and feature names
    Output: the loaded model, feature names and scalar
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError('No checkpoint found in candidates: ' + model_file)

    # Handle scikit-learn joblib files
    if model_file.endswith('.joblib'):
        import joblib
        ckpt = joblib.load(model_file)
        if isinstance(ckpt, dict):
            feature_names = ckpt["feature_names"]
            scaler = ckpt.get("scaler")
            model = ckpt["model"]
        else:
            # ckpt is the model itself; try to extract metadata
            model = ckpt
            feature_names = getattr(model, 'feature_names_in_', None)
            if feature_names is None:
                feature_names = getattr(model, 'feature_names_', None)
            scaler = None
            if feature_names is None:
                raise ValueError("joblib model has no feature_names metadata")
        return model, feature_names, scaler

    # Handle PyTorch checkpoints
    ckpt = torch.load(model_file, map_location=device, weights_only=False)

    feature_names: List[str] = ckpt["feature_names"]
    scaler = ckpt["scaler"]
    input_dim = int(ckpt["input_dim"])

    model = AerisFullStructure(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, feature_names, scaler

# -----------------------------
# Build X,y in *checkpoint feature order*
# -----------------------------
def _make_magpie_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer([
        cf.Stoichiometry(),
        cf.ElementProperty.from_preset("magpie"),
        cf.ValenceOrbital(props=['avg']),
        cf.IonProperty(fast=True),
    ])

def _compute_magpie_df(compositions: pd.Series) -> pd.DataFrame:
    featurizer = _make_magpie_featurizer()

    comp_objs = []
    for s in compositions.astype(str).tolist():
        try:
            comp_objs.append(Composition(s))
        except Exception:
            comp_objs.append(None)

    base = pd.DataFrame({"comp_obj": comp_objs}, index=compositions.index)

    try:
        feat_df = featurizer.featurize_dataframe(
            base, col_id="comp_obj", ignore_errors=True, pbar=False, n_jobs=1
        )
    except TypeError:
        try:
            featurizer.set_n_jobs(1)
        except Exception:
            pass
        feat_df = featurizer.featurize_dataframe(
            base, col_id="comp_obj", ignore_errors=True, pbar=False
        )

    feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c == "comp_obj"], errors="ignore")
    return feat_df

def build_features_in_ckpt_order(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Function to build the feature list in the order provided by the model.
    Input: - the dataframe with the raw data
           - the list of features needed by the model
           - the column that is predicted
    Output: - List of input features that can be used by the model to predict
            - Optionally, a list of output ground truth values for the predicted column
    """
    # optional numeric columns (if present in CSV) that we will include as features
    OPTIONAL_NUMERIC_COLS = [
        'density_atomic', 'CN_max', 'CN_min', 'CN_avg',
    ]

    required = ["composition", "structure"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'")

    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Missing target column '{target_col}'")

    magpie_df = _compute_magpie_df(df["composition"])
    n = len(df)
    X = np.zeros((n, len(feature_names)), dtype=np.float32)
    y: Optional[np.ndarray] = None
    if target_col is not None:
        y = np.zeros((n, 1), dtype=np.float32)

    df2 = df.reset_index(drop=True)

    for i, row in df2.iterrows():
        comp_str = str(row["composition"])

        # 1) element fractions
        try:
            parsed = parse_formula(comp_str)
            total = float(sum(parsed.values())) if parsed else 0.0
        except Exception:
            parsed, total = {}, 0.0

        elem_frac: Dict[str, float] = {}
        if total > 0:
            for el, cnt in parsed.items():
                elem_frac[el] = float(cnt) / total

        # 2) structure features
        struct_vals = parse_structure_string(row.get("structure"))

        # 3) optional numeric cols
        opt_vals: Dict[str, float] = {}
        for c in OPTIONAL_NUMERIC_COLS:
            if c in df2.columns:
                v = row.get(c)
                try:
                    opt_vals[c] = float(v)
                except Exception:
                    opt_vals[c] = np.nan

        # 4) magpie row
        magpie_row = magpie_df.iloc[i].to_dict()

        # single lookup dict, then assemble in EXACT feature_names order
        value_by_name: Dict[str, float] = {}
        for el, frac in elem_frac.items():
            value_by_name[el] = float(frac)
        for k, v in struct_vals.items():
            try:
                value_by_name[k] = float(v)
            except Exception:
                pass
        for k, v in opt_vals.items():
            try:
                value_by_name[k] = float(v)
            except Exception:
                pass
        for k, v in magpie_row.items():
            try:
                value_by_name[k] = float(v)
            except Exception:
                pass

        X[i, :] = np.array([value_by_name.get(name, 0.0) for name in feature_names], dtype=np.float32)

        if y is not None:
            try:
                y[i, 0] = float(row[target_col])  # type: ignore[arg-type]
            except Exception:
                y[i, 0] = np.nan

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    if y is not None:
        y = y.astype(np.float32)
        # drop rows where y is nan
        mask = ~np.isnan(y[:, 0])
        X = X[mask]
        y = y[mask]
    return X, y

def predict_energy(model_path: str, features: Dict[str, Any], target_col: Optional[str] = None) -> Dict[str, Any]:
    """Function to predict enthalpy using the model
    Input: - The path to the model
           - Dictionary with the input data for prediction
           - Optionally the target column in case accuracy needs to be computed
    Output: dictionary with the resulted energy
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if features is None:
        raise ValueError("features must be provided")
    df = pd.DataFrame([features])

    model_obj, feature_names, scaler = load_structure_model(model_path, device=device)
    X_raw, y = build_features_in_ckpt_order(df, feature_names=feature_names, target_col=target_col)

    # scale (must match training)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    ds = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    model_obj.eval()
    for (xb,) in loader: #TODO need to keep all entries in the batch not just the last
        xb = xb.to(device)
        per_atom = model_obj(xb).detach().cpu().numpy()  # (B,1)

    parsed = parse_formula(features['composition'])
    n_atoms = float(sum(parsed.values()))
    total_e = per_atom * n_atoms
    return {
        'per_atom_eV_per_atom': per_atom,
        'total_eV': total_e,
        'parsed_composition': parsed,
        'n_atoms': n_atoms,
        'device' : device
    }

def find_structure_in_datasets(df: pd.DataFrame, composition: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Function to find the structures in the database that correspond to a given composition
    Input: - dataframe with the dataset
           - the composition string
           - Optionally the max number of entries returned
    Output: dictionary with all the entries found
    """
    all_entries = []
    query_comp = _apply_df_parse_formula_str(composition)

    if 'composition_reduced' not in df.columns or 'structure' not in df.columns or 'e_total' not in df.columns:
        return []

    print(f"Search in datasets through {len(df)} entries")

    comp_series = df["composition_reduced"].apply(_apply_df_parse_formula_str)
    # Search for all matching compositions
    results = df[comp_series == query_comp]

    for index, row in results.iterrows():
        candidate = {
            'composition': row['composition'],
            'structure': row['structure'],
            'total_eV': row['e_total'],
            'per_atom_eV_per_atom': row['formation_energy_per_atom'],
            'spacegroup_number': row['spacegroup_number'],
            'density_atomic': row['density_atomic'],
            'CN_max': row['CN_max'],
            'CN_min': row['CN_min'],
            'CN_avg': row['CN_avg'],
            "source_file": 'dataset'
        }
        all_entries.append(candidate)

    # Respect the optional limit if provided
    if limit is not None and limit > 0 and len(all_entries) >= limit:
        return all_entries[:limit]

    return all_entries

def find_structure_templates(df: pd.DataFrame, composition: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Function to find the structure candidates in the database that correspond to a given composition
    Input: - dataframe with the dataset
           - the composition string
           - Optionally the max number of entries returned
    Output: dictionary with all the candidate entries
    """
    all_entries = []
    query_comp = parse_formula(composition)
    nelem_query = sum(set(query_comp.values()))
    if nelem_query == 0:
        return all_entries

    # Check if required columns exist
    if 'composition_reduced' not in df.columns or 'structure' not in df.columns:
        return []

    nelem_series = df["composition_reduced"].apply(_apply_df_parse_formula_num)
    # Search for all matching compositions
    results = df[(nelem_series % nelem_query == 0) |
                 (nelem_query % nelem_series == 0)]

    for index, row in results.iterrows():
        candidate = {
            'composition': row['composition'],
            'structure': row['structure'],
            'spacegroup_number': row['spacegroup_number'],
            'density_atomic': row['density_atomic'],
            'CN_max': row['CN_max'],
            'CN_min': row['CN_min'],
            'CN_avg': row['CN_avg'],
            "source_file": 'dataset'
        }
        all_entries.append(candidate)

    # Respect the optional limit if provided
    if limit is not None and limit > 0 and len(all_entries) >= limit:
        return all_entries[:limit]

    return all_entries


def _chunk_bounds(n, size, rank):
     q, r = divmod(n, size)
     start = rank * q + min(rank, r)
     stop = start + q + (1 if rank < r else 0)
     return start, stop

def find_energy_forall_structures(dataset_files: str, composition: str, model_path: str,
                                  only_matches: bool=False, comm=None, limit: Optional[int] = 100):
    """Function to find and/or predict the energy for a list of sructures
    Input: - dataframe with the dataset
           - the composition string
           - the path to the model
           - MPI communicator
           - Optionally the max number of entries returned
    Output: dictionary with all the entries found
    """

    rank = 0
    size = 1

    if AERIS_HAS_MPI:
        if comm is None:
            comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    if rank == 0:
        df = read_datasets(dataset_files)
        # find all the entries in the datasets given a composition
        entries = find_structure_in_datasets(df, composition)
        parsed = parse_formula(composition)
        comp_name = ''.join(f"{k}{v}" for k, v in parsed.items())
        n_atoms = float(sum(parsed.values()))
        # add all the candidates from the dataset that have energy recorded
        candidates = [{
            'per_atom_eV_per_atom': i["per_atom_eV_per_atom"],
            'total_eV': i["total_eV"],
            'parsed_composition': parsed,
            'n_atoms': n_atoms,
            'source': "dataset",
        } for i in entries if not np.isnan(i["per_atom_eV_per_atom"])]
        print(f"[{rank}] {len(candidates)} candidates found in the datasets")
        json.dump(candidates, sys.stdout, indent=2)
        print()

        f = open(f"{comp_name}.json", "w")
        for candidate in candidates:
            f.write(json.dumps(candidate) + "\n")

    if only_matches:
        return

    if rank == 0:
        # find all possible template compositions from the dataset
        template_entries = find_structure_templates(df, composition)
        print("Templates found:", len(template_entries))
        print(f"Distributing the templates to {size} ranks")
        n = len(template_entries)
    else:
        candidates = []
        template_entries = None
        n = None

    start = 0
    stop = n
    if AERIS_HAS_MPI:
        # Broadcast shared inputs
        n = comm.bcast(n, root=0)
        template_entries = comm.bcast(template_entries, root=0)
        start, stop = _chunk_bounds(n, size, rank)
        print(f"[{rank}] Search templates between {start} and {stop}")
    local_templates = template_entries[start:stop]

    for template in local_templates:
        parsed = parse_formula(template['composition'])
        template["composition"] = composition
        out = predict_energy(model_path, template)
        out['parsed_composition'] = parsed
        out['n_atoms'] = float(sum(parsed.values()))
        out['source'] = "predicted"
        candidates.append(out)

    if AERIS_HAS_MPI:
        all_candidates = comm.gather(candidates, root=0)
    else:
        all_candidates = candidates

    if rank == 0:
        for out in all_candidates:
            f.write(json.dumps(out) + "\n")
        f.close()
        print(f"Candidates for {composition} written to {comp_name}.json")


def _group_of_rank(rank):
    # first r groups get (q+1) ranks, rest get q ranks
    cutoff = (q + 1) * r
    if rank < cutoff:
        return rank // (q + 1)
    else:
        return r + (rank - cutoff) // q


def predict_all(dataset: str, model: str, limit: int = 10):
    df = read_datasets(dataset)
    print("Loaded dataframe:", df.shape)
    target_col = 'formation_energy_per_atom'
    for index, row in df.iterrows():
        candidate = {
            'composition': row['composition'],
            'structure': row['structure'],
            'total_eV': row['e_total'],
            'per_atom_eV_per_atom': row['formation_energy_per_atom'],
            'spacegroup_number': row['spacegroup_number'],
            'density_atomic': row['density_atomic'],
            'CN_max': row['CN_max'],
            'CN_min': row['CN_min'],
            'CN_avg': row['CN_avg'],
            "source_file": 'dataset',
            target_col: row[target_col]
        }
        y = row[target_col]
        pred = predict_energy(model, candidate, target_col)
        err = pred['per_atom_eV_per_atom'] - y
        print(f"Predicted per-atom energy: {pred['per_atom_eV_per_atom']} eV/atom, ground truth: {y} eV/atom")
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        print(f"Dataset metrics: MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")
        if index > limit:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility codes to find viable structures for a given composition')
    parser.add_argument('--nelements', type=int, default=2, help='Max atoms in the composition (e.g. 3 for U2O UO and UO2)')
    parser.add_argument('--only-matches', action='store_true', help='Look only for matches in the datasets')
    parser.add_argument('--limit', type=int, default=10, help='Limit the number of predictions')
    parser.add_argument('--predict-dataset', action='store_true', help='Predict on the entire dataset instead of searching for templates')
    args = parser.parse_args()

    DATASETS = 'Dataset*.csv'
    model_path = 'model/aeris_model.pt'
    if args.predict_dataset:
        predict_all(DATASETS, model_path, limit=args.limit)
        exit(0)

    world_rank = 0
    world_size = 1
    if AERIS_HAS_MPI:
        world = MPI.COMM_WORLD
        world_rank = world.Get_rank()
        world_size = world.Get_size()

    # look at all combinations of elements A and B with a given max number of atoms
    max_elements = int(args.nelements)
    unique_ratios = {}
    for a in range(1, max_elements + 1):
        for b in range(1, max_elements + 1):
            total = a + b
            ratio = a / total
            # Reduce the fraction to find the "minimum" A and B
            common = math.gcd(a, b)
            min_a, min_b = a // common, b // common
            # Store in dictionary to keep only unique ratios
            if (min_a, min_b) not in unique_ratios:
                unique_ratios[(min_a, min_b)] = ratio

    # Sort results by the ratio value
    sorted_results = sorted(unique_ratios.items(), key=lambda x: x[1])
    compositions = [f"U{ma}Si{mb}" for (ma, mb), _r in sorted_results]
    ncomp = len(compositions)

    if world_rank == 0:
        print(f"Looking at {ncomp} ratios of Si and U divided on {world_size} ranks")

    if not AERIS_HAS_MPI:
        for comp in compositions:
            find_energy_forall_structures(
                DATASETS,
                comp,
                model_path,
                only_matches=args.only_matches
            )
    elif world_size <= ncomp: # outer only (each rank takes some compositions), no inner MPI
        local_comps = compositions[world_rank::world_size]
        print(f"[{world_rank}] Looking at {len(local_comps)} compositions")
        for comp in local_comps:
            find_energy_forall_structures(
                DATASETS, comp, model_path,
                only_matches=args.only_matches,
                comm=MPI.COMM_SELF, # no parallelism
            )
    else: # enought ranks for both outer and inner parallelism, many ranks per each composition
        ngroups = ncomp
        # assign each world rank to a group (balanced)
        # ranks_per_group = ~world_size/ngroups; use a block partition
        q, r = divmod(world_size, ngroups)
        color = _group_of_rank(world_rank)  # group id = composition index
        subcomm = world.Split(color=color, key=world_rank)

        comp = compositions[color]  # all ranks in subcomm agree
        sub_rank = subcomm.Get_rank()

        # only subcomm root gets results; guard printing/writing
        if sub_rank == 0:
            print(f"[group {color} root world_rank {world_rank}] Looking at composition {comp}")

        # inner template split ON (across subcomm)
        find_energy_forall_structures(
            DATASETS, comp, model_path,
            only_matches=args.only_matches,
            limit=100,
            comm=subcomm,
        )
