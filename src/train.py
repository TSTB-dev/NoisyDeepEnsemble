import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import copy

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from tqdm import tqdm

from models import resnet18, efficientnetb0, vgg16
from dataset import build_dataset
from noise_injection import inject_noise

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()

def get_dataloader(dataset, rank, world_size, batch_size, num_workers):
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return loader

def get_scheduler(optimizer, epochs, scheduler_name):
    if scheduler_name == "cosine":
        print("Using cosine annealing scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        print("Using step scheduler")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    return scheduler

def get_optimizer(model, optimizer_name, lr, momentum=0.9, weight_decay=5e-4):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        params_groups = [
            {
                "params": (p for n, p in model.named_parameters() if ("bias" not in n) and ("bn" not in n)),
            }, {
                "params": (p for n, p in model.named_parameters() if ("bias" in n) or ("bn" in n)),
                "weight_decay": 0,
                "WD_exclude": True,
            }
        ]
        optimizer = optim.AdamW(params_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name")
    return optimizer
    
def train(rank, world_size):
    print(f"Running DDP with rank {rank} and world size {world_size}")
    parser = ArgumentParser()
    setup(rank, world_size)
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data_dir", type=str, default="data")
    
    # Training settings for the model
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    
    # Ensemble settings [train]
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--ensemble_type", type=str, default="noisy")  # "noisy" or "standard"
    parser.add_argument("--perturbation", type=str, default="uniform")  # "uniform" or "normal" or "none"
    parser.add_argument("--perturbation_type", type=str, default="additive")
    parser.add_argument("--reset_lr_scheduler", action="store_true")
    parser.add_argument("--perturbation_strength", type=float, default=0.1)
    parser.add_argument("--perturbation_ratio", type=float, default=0.1)
    parser.add_argument("--perturbation_mean", type=float, default=0.0)
    # parser.add_argument("--perturbation_scheduler", type=str, default=None)
    # parser.add_argument("--perturbation_epochs", type=int, default=10)
    # parser.add_argument("--perturbation_decay", type=float, default=0.1)
    parser.add_argument("--aux_epochs", type=int, default=10)
    
    # Ensemble settings [test]
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--kl_divergence", action="store_true")
    parser.add_argument("--disagreement", action="store_true")
    parser.add_argument("--ensemble_path_list", type=str, nargs="+", default=None)
    parser.add_argument("--ensemble_aggregation", type=str, default="mean")
    
    # Additional settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="models")
    
    args = parser.parse_args()
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        print(f"Save directory {args.save_dir} already exists")

    # Build dataset
    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)

    # Build dataloaders
    train_loader = get_dataloader(train_dataset, rank=rank, world_size=world_size, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = get_dataloader(test_dataset, rank=rank, world_size=world_size, batch_size=1, num_workers=args.num_workers)
    
    # Build model
    if args.dataset == "mnist":
        num_classes = 10
    elif args.dataset == "cifar10": 
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Invalid dataset name")
    if args.model == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif args.model == "efficientnet_b0":
        model = efficientnetb0(num_classes=num_classes)
    elif args.model == "vgg16":
        model = vgg16(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
        
    if args.ensemble_type == "standard":
        # Train standard model
        print(f"Training standard model for {args.epochs}...")
        ensemble_models = [copy.deepcopy(model) for _ in range(args.ensemble_size)]
        for i in range(args.ensemble_size):
            model = ensemble_models[i]
            print(f"Training model {i+1}... for {args.epochs} epochs")
            model.train()
            model.to(rank)
            ddp_model = DDP(model, device_ids=[rank])
            optimizer = get_optimizer(ddp_model, args.optimizer, args.lr)
            scheduler = get_scheduler(optimizer, args.epochs, args.scheduler)
            
            train_one_model(ddp_model, train_loader, test_loader, criterion, optimizer, scheduler, args.epochs, rank, args)
            torch.save(ddp_model.state_dict(), os.path.join(args.save_dir, f"{args.model}_{args.dataset}_{i}.pth"))
            print(f"Model {i+1} saved at {os.path.join(args.save_dir, f'{args.model}_{args.dataset}_{i}.pth')}")
            
        torch.distributed.barrier()
        # Test standard model
        test_results = test_ensemble(ensemble_models, test_loader, rank, args)
        print(test_results)
        
        return ddp_model
    
    # Train parent model
    model.train()
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = get_optimizer(ddp_model, args.optimizer, args.lr)
    scheduler = get_scheduler(optimizer, args.epochs, args.scheduler)
    
    print(f"Training parent model for {args.epochs}...")
    train_one_model(ddp_model, train_loader, test_loader, criterion, optimizer, scheduler, args.epochs, rank, args)
    # Save parent model
    torch.save(ddp_model.state_dict(), os.path.join(args.save_dir, f"{args.model}_{args.dataset}_parent.pth"))
    
    # Train child models
    models = {
        "parent": ddp_model,
        "ensemble": []
    }
    for i in range(args.ensemble_size):
        base_model = copy.deepcopy(models["parent"])
        base_model.to(rank)
        base_model.train()
        base_model_ddp = DDP(base_model, device_ids=[rank])
        optimizer = get_optimizer(base_model_ddp, args.optimizer, args.lr)
        scheduler = get_scheduler(optimizer, args.epochs, args.scheduler)
        
        print(f"Training model {i+1}...")
        
        if args.perturbation == "none":
            print(f"Performing no perturbation for model {i+1}")
            print(f"Copying parent model for model {i+1}")
            child_model = base_model_ddp
        elif args.perturbation == "uniform":
            print(f"Injecting uniform noise for model {i+1}")
            print(f"Strength: {args.perturbation_strength}")
            print(f"Ratio: {args.perturbation_ratio}")
            child_model = inject_noise(base_model_ddp, args.perturbation, args.perturbation_strength, args.perturbation_ratio, args.perturbation_mean)
        elif args.perturbation == "normal":
            print(f"Injecting normal noise for model {i+1}")
            print(f"Strength: {args.perturbation_strength}")
            print(f"Mean: {args.perturbation_mean}")    
            print(f"Ratio: {args.perturbation_ratio}")
            child_model = inject_noise(base_model_ddp, args.perturbation, args.perturbation_strength, args.perturbation_ratio, args.perturbation_mean)
            
        child_model.train()
        train_one_model(child_model, train_loader, test_loader, criterion, optimizer, scheduler, args.aux_epochs, rank, args)
        models["ensemble"].append(child_model)
        torch.save(child_model.state_dict(), os.path.join(args.save_dir, f"{args.model}_{args.dataset}_{i}.pth"))
        print(f"Model {i+1} saved at {os.path.join(args.save_dir, f'{args.model}_{args.dataset}_{i}.pth')}")
        
    # Test ensemble
    test_results = test_ensemble(models["ensemble"], test_loader, rank, args)
    print(test_results)
    
    cleanup()
    return models

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epochs: int,
    rank: int,
    args=None,
):  
    for epoch in range(epochs):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, args.log_interval, rank)
        
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_{args.dataset}_{epoch}.pth"))
            
        if epoch % args.log_interval == 0:
            test_one_model(model, train_loader, rank, "train", args)
            test_one_model(model, test_loader, rank, "test", args)
    
    # Test
    test_one_model(model, train_loader, rank, "train", args)
    test_one_model(model, test_loader, rank, "test", args)
    model.to("cpu")

@torch.no_grad()
def test_one_model(
    model: nn.Module,
    dataloader: DataLoader,
    rank: int,
    split: str,
    args=None,
):
    if split == "train":
        batch_size = args.batch_size
    else:
        batch_size = 1
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Testing"):
        images = batch["image"].to(rank)
        labels = batch["label"].to(rank)
        if len(images) != batch_size:
            print(f"Batch size mismatch: {len(images)} v.s. {batch_size}")
            continue
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    torch.distributed.barrier()
    print(f"[{split}] Accuracy: {correct / total}, Total: {total}, Correct: {correct}")
    
    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct
    }
    
def train_one_epoch(
    epoch: int, 
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    log_interval: int,
    rank: int,
    args=None,
):
    model.train()
    loss_train = 0
    acc_train = 0
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        images = batch["image"].to(rank)
        labels = batch["label"].to(rank)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_train += loss.item()
        acc_train += (outputs.argmax(1) == labels).sum().item() / len(labels)
        loss.backward()
        
        optimizer.step()
        
    if scheduler is not None:
        scheduler.step()
        
        if i % log_interval == 0:
            # print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item()}")
            pass
    print(f"Epoch {epoch} | Loss: {loss_train / len(train_loader)} | Accuracy: {acc_train / len(train_loader)}")
    return {
        "epoch": epoch,
        "accuracy": acc_train / len(train_loader),
        "loss": loss_train / len(train_loader),
    }

@torch.no_grad()
def test_ensemble(
    ensemble_models: list,
    test_loader: DataLoader,
    rank: int,
    args=None,
):
    for model in ensemble_models:
        model.eval()
    
    preds = []
    gts = []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing Ensemble"):
        images = batch["image"].to(rank)
        labels = batch["label"].to(rank)
    
        outputs = []
        for model in ensemble_models:
            model.to(rank)
            outputs.append(model(images))
            model.to("cpu")
        outputs = torch.stack(outputs)
        
        if args.ensemble_aggregation == "mean":
            output = outputs.mean(dim=0)
        elif args.ensemble_aggregation == "max":
            output = outputs.max(dim=0)
        elif args.ensemble_aggregation == "vote":
            output = outputs.mode(dim=0)
        else:
            raise ValueError("Invalid ensemble aggregation method")
        
        preds.append(output)
        gts.append(labels)
    
    preds = torch.stack(preds)
    gts = torch.stack(gts)
    
    # Calculate accuracy
    correct = 0
    total = 0
    for i in range(len(preds)):
        _, predicted = torch.max(preds[i], 1)
        total += labels.size(0)
        correct += (predicted == gts[i]).sum().item()
    print(f"Accuracy: {correct / total}")
    print(f"Total: {total}")
    
    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
    }
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    

