import os
import torch
from torch.utils.data import Dataset

from pymatgen.core.structure import Structure

from p_tqdm import p_umap
import pandas as pd
import numpy as np


def type_coords(frac_coords, atomic_numbers):
    atomic_tensor = atomic_numbers.reshape(-1, 1)
    all_tensor = torch.cat([atomic_tensor, frac_coords], -1)
    coe = 1000*all_tensor[:, 0] + 100*all_tensor[:, 1] + 10*all_tensor[:, 2] + 1*all_tensor[:, 3]
    index_order = torch.argsort(coe)
    return index_order


class CrystalDataset(Dataset):
    def __init__(self, task, dataset, config):
        paths = [os.path.join('data', task, dataset, 'ft_'+str(i)+'.csv') for i in range(config.n_value)]
        self.k_value = config.k_value
        self.path_prepare = os.path.join('data', task, dataset, "prepared_more_"+str(config.n_value)+"_"+str(self.k_value)+"_data.pt")

        self.config = config
        self.num_workers = config.dataset.num_workers
        self.primitive = config.dataset.primitive

        self.read_from_cif(paths)
        self.offset = torch.cat([torch.tensor([0]), torch.cumsum(self.data["num_atoms"], -1)], 0).long()
        self.matrix_scaler = None
    
    def __len__(self) -> int:
        return len(self.data['material_id'])

    def __getitem__(self, index):
        start, end = self.offset[index], self.offset[index+1]
        scaled_matrix, num_atoms = self.data['scaled_matrix'][index], self.data['num_atoms'][index]
        frac_coords = self.data['frac_coords'][start: end]
        atomic_numbers = self.data['atomic_numbers'][start: end]
        scalar_matrix = self.matrix_scaler.transform(scaled_matrix)

        material_id = self.data["material_id"][index]

        return material_id, scalar_matrix, frac_coords, atomic_numbers, num_atoms
        
    def add_scaled_matrix(self, data, scale_len=True):
        matrix = data['matrix']
        num_atoms = data['num_atoms'].reshape(-1, 1)
        if scale_len:
            matrix = matrix / num_atoms.float()**(1/3)
        data['scaled_matrix'] = matrix

    def read_from_cif(self, paths):
        if not os.path.exists(self.path_prepare):
            df = [pd.read_csv(p) for p in paths]
            df = [one[:int(self.k_value*len(one))] for one in df]
            df = pd.concat(df, axis=0, ignore_index=True)

            unordered_results = np.array(p_umap(self.cif_info, [df.iloc[idx] for idx in range(len(df))], num_cpus=self.num_workers))
            order = np.array([result['material_id'] for result in unordered_results]).argsort()
            order_results = unordered_results[order]
            data = self.unpack(order_results)
            self.add_scaled_matrix(data)
            self.data = {
                'material_id': data['material_id'], 
                'frac_coords': data['frac_coords'], 
                'atomic_numbers': data['atomic_numbers'], 
                'scaled_matrix': data['scaled_matrix'],
                'num_atoms': data['num_atoms'],
                }
            torch.save(self.data, self.path_prepare)
        else:
            self.data = torch.load(self.path_prepare, weights_only=False)
    
    def unpack(self, results):
        material_id, frac_coords, atomic_numbers, matrix, num_atoms = [], [], [], [], []
        for re in results:
            material_id.append(re['material_id'])
            num_atoms.append(torch.LongTensor([re['num_atoms']]))
            matrix.append(re['matrix'])
            frac_coords.append(re['frac_coords'])
            atomic_numbers.append(re['atomic_numbers'].long())

        matrix, material_id, num_atoms, frac_coords, atomic_numbers = \
            torch.stack(matrix, 0), np.array(material_id), torch.cat(num_atoms, 0), torch.cat(frac_coords, 0), torch.cat(atomic_numbers, 0)
        return {'material_id': material_id, 'frac_coords':frac_coords, 'atomic_numbers':atomic_numbers, \
                'matrix':matrix, 'num_atoms': num_atoms}

    def cif_info(self, row):
        cif, material_id = row['cif'], row['material_id']
        structure = Structure.from_str(cif, fmt='cif')
        if self.primitive:
            structure = structure.get_primitive_structure()
        structure = structure.get_reduced_structure()  ## niggli
        
        frac_coords = torch.FloatTensor(structure.frac_coords)
        atomic_numbers = torch.FloatTensor(structure.atomic_numbers)
        index_order = type_coords(frac_coords, atomic_numbers)
        frac_coords = frac_coords[index_order]
        atomic_numbers = atomic_numbers[index_order]
        num_atom = len(atomic_numbers)

        matrix = torch.tensor(structure.lattice.matrix)
        matrix = self.compute_lattice_polar_decomposition(matrix)
        
        return {'material_id': material_id, 'frac_coords':frac_coords, 'atomic_numbers':atomic_numbers, \
                'matrix': matrix, 'num_atoms': num_atom}

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
