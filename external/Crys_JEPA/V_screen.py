import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from components.jepa.frame.jepa import JEPA
from utils.utils import parse_args_and_config, get_scaler_mean_std, last_ckpt

from itertools import combinations
from tqdm import tqdm

def expand_into_subsystems(chemical_system: str) -> list[tuple[str, ...]]:
    elements = chemical_system.split('-')
    list_combinations = []
    for n in range(1, len(elements) + 1):
        list_combinations += list(combinations(elements, n))  ## C_{elements}^n
    return list_combinations

def compute_lattice_polar_decomposition(lattice_matrix: torch.Tensor) -> torch.Tensor:
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

def deal_structures(structures, matrix_scaler):
    matrix, frac_coords, atomic_numbers, num_atoms = [], [], [], []
    for stru in structures:
        frac_coords.append(torch.FloatTensor(stru.frac_coords))
        atomic_numbers.append(torch.LongTensor(stru.atomic_numbers))
        num_atoms.append(len(stru.atomic_numbers))
        matrix.append(compute_lattice_polar_decomposition(torch.FloatTensor(stru.lattice.matrix)))
    matrix = torch.stack(matrix, 0)
    frac_coords = torch.cat(frac_coords, 0)
    atomic_numbers = torch.cat(atomic_numbers, 0)
    num_atoms = torch.LongTensor(num_atoms)

    scaled_matrix = matrix / num_atoms.unsqueeze(1).float()**(1/3)
    scalar_matrix = matrix_scaler.transform(scaled_matrix)
    return scalar_matrix.float().cuda(), frac_coords.float().cuda(), atomic_numbers.cuda(), num_atoms.cuda()

def get_emb(data, model):
    matrix, frac_coords, atomic_numbers, num_atoms = data
    b = matrix.shape[0]
    batch = torch.repeat_interleave(torch.arange(b).to(matrix.device), num_atoms)
    
    x = torch.cat([frac_coords, F.one_hot(atomic_numbers, 100).float(), matrix[batch]], -1)
    cls_out = model.encode(x, batch)
    return cls_out.detach().cpu()


args, config = parse_args_and_config('jepa')
if args.dataset == "mp_20":
    num_turns = 40
    indicators_mattersim = "indicators_mattersim_mp_20.pt"
elif args.dataset == "alex_mp_20":
    num_turns = 10
    indicators_mattersim = "indicators_mattersim_alex_mp_20.pt"

matrix_scaler = get_scaler_mean_std(args.task)
model = JEPA(config, matrix_scaler=matrix_scaler)
ckpt = last_ckpt('jepa')
checkpoint = torch.load(ckpt)
new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(new_state_dict)
model.eval()
model = model.cuda()

dis_path = os.path.join('./logs/base/', args.dataset, 'dis')
os.makedirs(dis_path, exist_ok=True)
ft_data_path = os.path.join('./data/finetune', args.dataset)
os.makedirs(ft_data_path, exist_ok=True)

ref = pd.read_csv('./data/base/'+args.dataset+'/train.csv.gz', compression='gzip')
eval_path = os.path.join('./logs/base/', args.dataset, 'eval')
for turn in range(num_turns):
    structures, vun, cifs = [], [], []
    for p in range(10):
        try:
            path_relax = pd.read_json(os.path.join(eval_path, str(turn*10+p)+'/relaxed_mattersim.json'))
            for s in path_relax['structure']:
                structures.append(Structure.from_dict(s))
                cifs.append(CifWriter(Structure.from_dict(s)).__str__())

            path_indicators = os.path.join(eval_path, str(turn*10+p), indicators_mattersim)
            indicators = torch.load(path_indicators, weights_only=False)
            vun.append(indicators['vu'] * indicators['vn'])
        except:
            pass
    vun = np.hstack(vun)
    assert len(structures) == len(vun)

    emb_dis = []
    for s_gen in tqdm(structures):
        curr_structures = []
        cs_gen = s_gen.composition.chemical_system
        curr_structures.append(s_gen)
        
        subsys = expand_into_subsystems(cs_gen)
        for s in subsys:
            for key in ['-'.join(sorted(s))]:
                ref_rows = ref[ref['chemical_system']==key]
                if len(ref_rows) > 0:
                    ref_cifs = ref_rows['cif'].values
                    for cif in ref_cifs:
                        s_ref = Structure.from_str(cif, fmt='cif')
                        curr_structures.append(s_ref)
        
        if len(curr_structures) > 1:
            data = deal_structures(curr_structures, matrix_scaler)
            emb = get_emb(data, model).data.cpu()
            ref_emb = emb[1:]
            gen_emb = emb[0].unsqueeze(0).repeat(ref_emb.shape[0], 1)
            emb_dis.append(((gen_emb - ref_emb)**2).sum(-1).mean().item())
        else:
            emb_dis.append(1e9)
    emb_dis = torch.FloatTensor(emb_dis)
    torch.save(emb_dis, os.path.join(dis_path, 'dis_'+str(turn)+'.pt'))
    
    index = torch.arange(len(emb_dis))
    vun_dis = emb_dis[vun]
    vun_index = index[vun]

    vun_order = np.argsort(vun_dis)
    vun_index_order = vun_index[vun_order]

    vun_order_cifs = [cifs[i] for i in vun_index_order]
    rows = []
    for i, cif in enumerate(vun_order_cifs):
        one = {}
        one['material_id'] = str(i)
        one['cif'] = cif
        rows.append(one)
    df = pd.DataFrame(rows, columns=['material_id', 'cif'])
    df.to_csv(os.path.join(ft_data_path, 'ft_'+str(turn)+'.csv'), index=False)
