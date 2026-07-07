import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from components.jepa.dataloader import CrystalDataset
from components.jepa.frame.jepa import JEPA
from utils.utils import parse_args_and_config, get_scaler_mean_std, check_save_num

import wandb
import warnings
from tqdm import tqdm


warnings.filterwarnings('ignore')

def collate(batch):
    frac_coords, matrix, atomic_numbers, ori_matrix, num_atoms, ef_per_atom = zip(*batch)
    
    frac_coords = torch.cat(frac_coords, 0).float()
    matrix = torch.stack(matrix, 0).float()
    atomic_numbers = torch.cat(atomic_numbers, 0).long()
    ori_matrix = torch.stack(ori_matrix, 0)

    num_atoms = torch.LongTensor(list(num_atoms))
    ef_per_atom = torch.FloatTensor(list(ef_per_atom))
    return frac_coords, matrix, atomic_numbers, ori_matrix, num_atoms, ef_per_atom

def train(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    ## Initialize Dataset
    train = CrystalDataset(config)
    scaled_matrix = train.data['scaled_matrix']
    train.matrix_scaler = get_scaler_mean_std(args.task, scaled_matrix)
    print("Size of Training set: {}".format(train.__len__()))

    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, sampler=train_sampler, batch_size=config.training.batch_size, collate_fn=collate)
    

    ## Initialize Model
    model = JEPA(config, matrix_scaler=train.matrix_scaler).to(device)
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))

    model = DDP(model, device_ids=[rank], output_device=rank)

    ## Initialize Wandb
    if rank==0:
            wandb.init(
                project=config.wandb.project, 
                name=args.task, 
                config=config, 
                resume="allow"
            )

    ## Initialize Optimizer and Scheduler
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
    for epoch in tqdm(range(start_epoch+1, start_epoch+config.training.epochs), desc="Training..."):
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
    args, config = parse_args_and_config("jepa")
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, config), nprocs=world_size, join=True)