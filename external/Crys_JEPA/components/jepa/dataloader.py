import os
import torch
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure

from utils.crys_utils import vector2matrix

from p_tqdm import p_umap
import pandas as pd
import numpy as np


def type_coords(coords, atomic_numbers):
    atomic_tensor = atomic_numbers.reshape(-1, 1)
    all_tensor = torch.cat([atomic_tensor, coords], -1)
    coe = 1000*all_tensor[:, 0] + 100*all_tensor[:, 1] + 10*all_tensor[:, 2] + 1*all_tensor[:, 3]
    index_order = torch.argsort(coe)
    return index_order

class CrystalDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.fetch_datasets()
        self.matrix_scaler = None
        self.offset = torch.cat([torch.tensor([0]), torch.cumsum(self.data["num_atoms"], -1)], 0).long()

    def __len__(self) -> int:
        return len(self.data['num_atoms'])

    def __getitem__(self, index):
        start, end = self.offset[index], self.offset[index+1]
        matrix, scaled_matrix, num_atoms, ef_per_atom = \
            self.data['matrix'][index], self.data['scaled_matrix'][index], self.data['num_atoms'][index], self.data['ef_per_atom'][index]
    
        frac_coords = self.data['frac_coords'][start: end]
        atomic_numbers = self.data['atomic_numbers'][start: end]
        ori_matrix = vector2matrix(matrix)

        scalar_matrix = self.matrix_scaler.transform(scaled_matrix)
        return frac_coords, scalar_matrix, atomic_numbers, ori_matrix, num_atoms, ef_per_atom

    def add_scaled_matrix(self, data):
        matrix = data['matrix']
        num_atoms = data['num_atoms'].reshape(-1, 1)
        matrix = matrix / num_atoms.float()**(1/3)
        data['scaled_matrix'] = matrix

    def fetch_datasets(self):
        data = []
        print("Loading mp2023")
        mp2023_path = os.path.join('data', "jepa", "prepared_mp2023_sub.pt")
        if os.path.exists(mp2023_path):
            mp2023 = torch.load(mp2023_path, weights_only=False)
        else:
            mp2023 = self.read_from_csv("./data/jepa/mp.csv.gz")
            torch.save(mp2023, mp2023_path)
        data.append(mp2023)
        
        print("Loading mptrj")
        mptrj_path = os.path.join('data', "jepa", "prepared_mptrj_sub.pt")
        if os.path.exists(mptrj_path):
            mptrj = torch.load(mptrj_path, weights_only=False)
        else:
            mptrj = self.read_from_csv("./data/jepa/mptrj.csv.gz")
            torch.save(mptrj, mptrj_path)
        data.append(mptrj)
        self.data = {k: torch.cat([one[k] for one in data], 0) for k in data[0].keys() if k!='material_id'}
    
    def read_from_csv(self, path):
        data = pd.read_csv(path, compression="gzip")
        unordered_results = np.array(p_umap(self.cif_info, [data.iloc[idx] for idx in range(len(data))], num_cpus=10))
        order = np.array([result['material_id'] for result in unordered_results]).argsort()
        order_results = unordered_results[order]
        data = self.unpack(order_results)
        self.add_scaled_matrix(data)
        return data

    def unpack(self, results):
        material_id, frac_coords, atomic_numbers, matrix, ef_per_atom, num_atoms = [], [], [], [], [], []
        for re in results:
            material_id.append(re['material_id'])
            num_atoms.append(torch.LongTensor([re['num_atoms']]))
            matrix.append(re['matrix'])
            frac_coords.append(re['frac_coords'])
            ef_per_atom.append(torch.FloatTensor([re['ef_per_atom']]))
            atomic_numbers.append(re['atomic_numbers'].long())

        matrix, material_id, num_atoms, frac_coords, ef_per_atom, atomic_numbers = \
            torch.stack(matrix, 0), np.array(material_id), torch.cat(num_atoms, 0), torch.cat(frac_coords, 0), torch.cat(ef_per_atom, 0), torch.cat(atomic_numbers, 0)
        return {'material_id': material_id, 'frac_coords':frac_coords, 'atomic_numbers':atomic_numbers, \
                'matrix':matrix, 'ef_per_atom': ef_per_atom, 'num_atoms': num_atoms}
    
    def cif_info(self, row):
        cif, material_id, ef_per_atom = row['cif'], row["material_id"], row["ef_per_atom"]
        structure = Structure.from_str(cif, fmt='cif')
        structure = structure.get_primitive_structure()
        structure = structure.get_reduced_structure()
        
        frac_coords = torch.FloatTensor(structure.frac_coords)
        atomic_numbers = torch.FloatTensor(structure.atomic_numbers)
        index_order = type_coords(frac_coords, atomic_numbers)
        frac_coords = frac_coords[index_order]
        atomic_numbers = atomic_numbers[index_order]
        num_atom = len(atomic_numbers)

        matrix = torch.tensor(structure.lattice.matrix)
        sym_matrix = self.compute_lattice_polar_decomposition(matrix)
        
        return {'material_id':material_id, 'frac_coords':frac_coords, 'atomic_numbers':atomic_numbers, \
                'matrix': sym_matrix, "ef_per_atom": ef_per_atom, 'num_atoms': num_atom}
    
    def compute_lattice_polar_decomposition(self, lattice_matrix: torch.Tensor) -> torch.Tensor:
        W, S, V_transp = torch.linalg.svd(lattice_matrix)
        S_square = torch.diag_embed(S)
        V = V_transp.transpose(0, 1)
        U = W @ V_transp
        P = V @ S_square @ V_transp
        P_prime = U @ P @ U.transpose(0, 1)
        # symmetrized lattice matrix
        symm_lattice_matrix = P_prime
        symm_lattice_matrix[torch.abs(symm_lattice_matrix) < 1e-5] = 0.

        tri_indices = torch.triu_indices(3, 3)
        symm_lattice_matrix = symm_lattice_matrix[tri_indices[0], tri_indices[1]]
        return symm_lattice_matrix
