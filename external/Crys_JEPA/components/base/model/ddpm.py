import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from components.base.model.type import type_noise
from components.base.model.matrix import matrix_noise
from components.base.model.frac import frac_noise
from components.base.model.transformer import Transformer

from tqdm import tqdm

class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.config = config
        self.h, self.w = config.dataset.max_atoms, config.dataset.max_types
        self.decoder = Transformer(config)
        self.T = config.diffusion.timesteps
        self.device = config.device
        self.type_noise, self.matrix_noise, self.frac_noise = type_noise(self.T), matrix_noise(self.T), frac_noise(self.T)

    def to_device(self, data):
        return [i.to(self.device) for i in data]
    
    def forward(self, data):
        matrix, frac_coords, atomic_numbers, num_atoms = self.to_device(data)
        b, device = matrix.shape[0], matrix.device
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)

        batch_frac_coords, _ = to_dense_batch(frac_coords, batch)
        batch_atomic_numbers, mask = to_dense_batch(F.one_hot(atomic_numbers, 100), batch)

        t = torch.randint(0, self.T, (b,), device=self.device).long()
        t_reshape = t.reshape(b, 1, 1)

        noisy_matrix, matrix_noise = self.matrix_noise(t_reshape, matrix.unsqueeze(1).repeat(1, batch_frac_coords.shape[1], 1))
        noisy_frac, frac_noise = self.frac_noise(t_reshape, batch_frac_coords)
        noisy_type, type_noise = self.type_noise(t_reshape, batch_atomic_numbers)

        x_t = self.compose(noisy_matrix, noisy_type, noisy_frac)
        pred_noise = self.decoder(x_t, mask, t)

        loss = self.cal_loss(pred_noise, matrix_noise, type_noise, frac_noise, mask)
        return loss

    def compose(self, matrix, type, frac):
        whole = torch.cat([frac, type, matrix], -1)
        return whole
    
    def reverse(self, b, num_atoms):
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        _, mask = to_dense_batch(batch, batch)
        max_atom = num_atoms.max()

        
        noisy_matrix = self.matrix_noise.gaussian(torch.Size([b, max_atom, 6]), self.device)
        noisy_frac = self.frac_noise.gaussian(torch.Size([b, max_atom, 3]), self.device)
        noisy_type = self.type_noise.gaussian(torch.Size([b, max_atom, self.w]), self.device)
        x = self.compose(noisy_matrix, noisy_type, noisy_frac)

        for t in tqdm(reversed(range(1, self.T)), desc='Sampling ...', total=self.T):
            curr_t = (torch.ones(b) * t).long().to(self.device)
            curr_t_reshape = curr_t.reshape(b, 1, 1)
            pred_noise = self.decoder(x, mask, curr_t)
            x = self.one_reverse(pred_noise, x, curr_t_reshape)
        return x[mask], batch

    def one_reverse(self, pred_noise, x, t):
        pred_matrix_noise, pred_type_noise, pred_frac_noise = pred_noise[:, :, -6:], pred_noise[:, :, 3:-6], pred_noise[:, :, :3]
        noisy_matrix, noisy_type, noisy_frac = x[:, :, -6:], x[:, :, 3:-6], x[:, :, :3]

        new_matrix = self.matrix_noise.reverse(t, noisy_matrix, pred_matrix_noise)
        new_type = self.type_noise.reverse(t, noisy_type, pred_type_noise)
        new_frac = self.frac_noise.reverse(t, noisy_frac, pred_frac_noise)
        new_x = self.compose(new_matrix, new_type, new_frac)
        return new_x   

    def cal_loss(self, pred_noise, matrix_noise, type_noise, frac_noise, mask):
        pred_matrix_noise, pred_type_noise, pred_frac_noise = pred_noise[:, :, -6:], pred_noise[:, :, 3:-6], pred_noise[:, :, :3]
        
        loss_matrix = self.matrix_noise.cal_loss(pred_matrix_noise[mask], matrix_noise[mask])
        loss_frac = self.frac_noise.cal_loss(pred_frac_noise[mask], frac_noise[mask])
        loss_type = self.type_noise.cal_loss(pred_type_noise[mask], type_noise[mask])
        return (loss_matrix + loss_type + loss_frac) / 3