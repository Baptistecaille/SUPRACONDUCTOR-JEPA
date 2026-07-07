import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from components.jepa.backbone.transformer import Transformer, MLP
from components.jepa.augmentation import rotate, translate


def batch_type_coords(coords, atomic_numbers):
    atomic_tensor = atomic_numbers.unsqueeze(-1)
    all_tensor = torch.cat([atomic_tensor, coords], -1)
    coe = 1000*all_tensor[:, :, 0] + 100*all_tensor[:, :, 1] + 10*all_tensor[:, :, 2] + 1*all_tensor[:, :, 3]
    batch_index_order = torch.argsort(coe)
    return batch_index_order
    
class JEPA(nn.Module):
    def __init__(self, config, matrix_scaler=None):
        super().__init__()
        self.pre_backbone = MLP(109, config.model.hidden_dim, config.model.hidden_dim)
        self.backbone = Transformer(config.model.hidden_dim, config.model.layers, config.model.attn_head, config.model.dropout)
        self.predictor = MLP(config.model.hidden_dim, config.model.hidden_dim, config.model.hidden_dim)
        self.cond_emb = MLP(6, config.model.hidden_dim, config.model.hidden_dim)
        self.device = config.device
        self.matrix_scaler = matrix_scaler

    def to_device(self, data):
        return [i.to(self.device) for i in data]
    
    def encode(self, x, batch):
        x, mask = to_dense_batch(x, batch)
        mask = torch.cat([torch.BoolTensor([[True]]).to(mask.device).repeat(mask.shape[0], 1), mask], 1) # one more True for cls_token
        x = self.pre_backbone(x)
        x = self.backbone(x, mask)
        cls_out = x[:, 0]
        return cls_out

    def cal_weight(self, ef_per_atom):
        weight = 1. - torch.exp(-torch.abs(ef_per_atom.unsqueeze(0) - ef_per_atom.unsqueeze(1)))
        return weight
    
    def get_loss(self, context, target, ef_per_atom):
        b = context.shape[0]
        
        context_norm = torch.norm(context, dim=-1, keepdim=True)
        target_norm = torch.norm(target, dim=-1, keepdim=True)
        dot_numerator = torch.mm(context, target.t())
        dot_denominator = torch.mm(context_norm, target_norm.t())
        sim = dot_numerator / dot_denominator
        
        weight = self.cal_weight(ef_per_atom.reshape(-1))
        diag = torch.eye(b, dtype=torch.bool).float().to(self.device)
        weight = weight + diag

        sim = torch.exp(sim * weight / 0.1)
        sim = sim / (sim.sum(-1).view(-1, 1) + 1e-8)
        return -torch.log(sim.diag() + 1e-8).mean()
    
    def forward(self, data):
        frac_coords, matrix, atomic_numbers, ori_matrix, num_atoms, ef_per_atom = self.to_device(data)
        frac_aug, matrix_aug, atomic_numbers_aug, aug_params = self.aug_batch(frac_coords, ori_matrix, atomic_numbers, num_atoms)
        
        b = matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        
        target = torch.cat([frac_coords, F.one_hot(atomic_numbers, 100).float(), matrix[batch]], -1)
        target = self.encode(target, batch)

        cond = self.cond_emb(aug_params)
        context = torch.cat([frac_aug.float(), F.one_hot(atomic_numbers_aug, 100).float(), matrix_aug[batch].float()], -1)
        context = self.encode(context, batch)
        context = self.predictor(context + cond)

        loss = self.get_loss(context, target, ef_per_atom)
        return loss

    def aug_batch(self, frac_coords, ori_matrix, atomic_numbers, num_atoms):
        b = ori_matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        
        ## aug trans
        frac_aug, vec_t = translate(frac_coords, b, batch)
        batch_frac_aug, mask = to_dense_batch(frac_aug, batch)
        batch_frac_aug[~mask] = 1000
        batch_atomic_numbers, _ = to_dense_batch(atomic_numbers, batch)
        batch_atomic_numbers[~mask] = 1000
        batch_index_order = batch_type_coords(batch_frac_aug, batch_atomic_numbers)

        batch_frac_aug = batch_frac_aug.gather(dim=1, index=batch_index_order.unsqueeze(-1).expand_as(batch_frac_aug))
        batch_atomic_numbers = batch_atomic_numbers.gather(dim=1, index=batch_index_order)
        frac_aug, atomic_numbers_aug = batch_frac_aug[mask], batch_atomic_numbers[mask]

        ## aug rot
        matrix_aug, vec_r = rotate(ori_matrix, b)

        tri_indices = torch.triu_indices(3, 3).to(self.device)
        matrix_aug = matrix_aug[:, tri_indices[0], tri_indices[1]]
        matrix_aug = matrix_aug / num_atoms.reshape(-1, 1)**(1/3)
        matrix_aug = self.matrix_scaler.transform(matrix_aug)

        ## aug params
        aug_params = torch.cat([vec_t, vec_r], -1)
        return frac_aug, matrix_aug, atomic_numbers_aug, aug_params

    def get_emb(self, data):
        frac_coords, matrix, atomic_numbers, _, num_atoms, ef_per_atom = self.to_device(data)
        b = matrix.shape[0]
        batch = torch.repeat_interleave(torch.arange(b).to(self.device), num_atoms)
        ori_x = torch.cat([frac_coords, F.one_hot(atomic_numbers, 100).float(), matrix[batch]], -1)
        backbone_out = self.encode(ori_x, batch)

        return backbone_out.data.cpu(), ef_per_atom.data.cpu()
