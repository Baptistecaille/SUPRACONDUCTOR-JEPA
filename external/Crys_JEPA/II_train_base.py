import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from components.base.model.ddpm import DDPM
from components.base.dataloader import CrystalDataset
from utils.utils import parse_args_and_config, get_scaler_min_max, last_ckpt, check_save_num

import warnings
import wandb
from tqdm import tqdm

warnings.filterwarnings('ignore')


def collate(batch):
    material_id, matrix, frac_coords, atomic_numbers, num_atoms = zip(*batch)  # unzip list of tuples
    matrix = torch.stack(matrix, 0)
    frac_coords = torch.cat(frac_coords, 0)
    atomic_numbers = torch.cat(atomic_numbers, 0)
    num_atoms = torch.LongTensor(list(num_atoms))
    return matrix.float(), frac_coords.float(), atomic_numbers.long(), num_atoms
    
def train(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    start_epoch = 0

    ## initialize
    train = CrystalDataset(args.task, args.dataset, config=config)
    scaled_matrix = train.data['scaled_matrix']
    train.matrix_scaler = get_scaler_min_max(args.task, args.dataset, scaled_matrix)
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, sampler=train_sampler, batch_size=config.training.batch_size, collate_fn=collate)

    device = torch.device(f"cuda:{rank}")
    model = DDPM(config).to(device)
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    model = DDP(model, device_ids=[rank], output_device=rank)
    if rank==0:
            wandb.init(
                project=config.wandb.project, 
                name=args.task+"_"+args.dataset,
                config=config,
                resume="allow"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=5e-2)
    if config.use_gradscalar:
        scaler = GradScaler(enabled = True)
    if config.use_schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=60, min_lr=1e-6)
    else:
        scheduler = None

    best_loss = 1e9
    start_epoch = 0
    print("start at ", start_epoch)

    ## Training
    best_loss = 1e9
    for epoch in tqdm(range(start_epoch+1, start_epoch+config.training.epoch), desc="Training..."):
        curr_loss = []
        train_sampler.set_epoch(epoch)
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(data).mean()
            if config.use_gradscalar:
                with autocast(device_type="cuda"):
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            curr_loss.append(loss.detach().item())
            
        curr_loss = sum(curr_loss) / len(curr_loss)
        if rank==0:
            wandb.log({"epoch": epoch, "epoch_loss": curr_loss}, step=epoch)
            if config.use_schedule:
                scheduler.step(curr_loss)
                wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
            
            if curr_loss < best_loss:
                best_loss = curr_loss
                save_path = os.path.join(args.log, 'saved_model')
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                }, os.path.join(save_path , f'model_{epoch}.pt'))
                check_save_num(save_path)
                
    args.logger.info('training completed')
    dist.destroy_process_group()
    if rank==0:
        wandb.finish()

if __name__ == '__main__':
    args, config = parse_args_and_config("base")
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, config), nprocs=world_size, join=True)