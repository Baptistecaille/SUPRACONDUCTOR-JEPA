
import os
import torch
from torch_scatter import scatter_mean

from components.base.dataloader import CrystalDataset
from components.base.model.ddpm import DDPM
from utils.crys_utils import vector2matrix
from utils.config import get_params, DictAction
from utils.utils import dict2namespace, last_ckpt, get_scaler_min_max

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mp_20')
parser.add_argument('--conf_new', nargs='+', action=DictAction)
args = parser.parse_args()
args.task = "base"
args.log = os.path.join('logs', args.task, args.dataset)
args.cfg = os.path.join(args.log, 'backup-config.yaml')
config = get_params(args, down=False)
config = dict2namespace(config)
device = torch.device('cuda')
config.device = device

diffusion = DDPM(config).cuda()
ckpt = last_ckpt(args.task, args.dataset)
checkpoint = torch.load(ckpt)
new_state_dict = {k.replace('module.', '', 1): v for k, v in checkpoint["model_state_dict"].items()}
diffusion.load_state_dict(new_state_dict)
diffusion.eval()

if args.dataset == "mp_20":
    num = 400_000
elif args.dataset == "alex_mp_20":
    num = 100_000
b = 1000
num_batch = int(num / b)
dataset = CrystalDataset(args.task, args.dataset, config=config)
num_prob = torch.zeros(21)
for i in range(1, 21):
    num_prob[i] = (dataset.data["num_atoms"] == i).sum()
num_prob /= num_prob.sum()

matrix_scaler = get_scaler_min_max(args.task, args.dataset)

saved_path = os.path.join(args.log, "gen")
os.makedirs(saved_path, exist_ok=True)
for i in tqdm(range(num_batch)):
    gen_crys = []
    num_atoms = torch.multinomial(num_prob, num_samples=b, replacement=True).cuda()
    with torch.no_grad():
        x, batch = diffusion.reverse(b, num_atoms)
        atomic_numbers_pred = x[:, 3:-6].argmax(-1)
        atomic_numbers_pred[atomic_numbers_pred==0] = 1
        frac_coords_pred = x[:, :3]
        frac_coords_pred[frac_coords_pred < 0.] = 0.
        frac_coords_pred[frac_coords_pred > 1.] = 1.

        matrix_pred = matrix_scaler.inverse_transform(scatter_mean(x[:, -6:], batch, dim=0)) * num_atoms.reshape(-1, 1).float()**(1/3)

        start_id = 0
        for idx_in_batch, num in enumerate(num_atoms):
            _atom_types = atomic_numbers_pred[start_id: start_id + num]
            _frac_coords = frac_coords_pred[start_id: start_id + num]
            _matrix = matrix_pred[idx_in_batch]

            gen_crys.append(
                {
                    "atom_types": _atom_types.detach().cpu().numpy(),
                    "frac_coords": _frac_coords.detach().cpu().numpy(),
                    "matrices": vector2matrix(_matrix.detach().cpu()).numpy()
                }
            )
            start_id = start_id + num
    torch.save(gen_crys, os.path.join(saved_path, "gen_"+str(i)+".pt"))
