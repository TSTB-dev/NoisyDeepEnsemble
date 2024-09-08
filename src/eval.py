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

def get_dataloader(dataset, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    return loader

def eval():
    print("Starting evaluation")
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    
    # Model
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--ckpt_dir", default="checkpoints", type=str)
    parser.add_argument("--ensemble_size", default=10, type=int)
    
    # Additional
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    
    args = parser.parse_args()
    
    # Build dataset
    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)
    
    # Build dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers)
    test_loader = get_dataloader(test_dataset, args.batch_size, args.num_workers)
    
    # Build model
    if args.model == "resnet18":
        model = resnet18(num_classes=train_dataset.num_classes)
    elif args.model == "vgg16":
        model = vgg16(num_classes=train_dataset.num_classes)
    elif args.model == "efficientnet_b0":
        model = efficientnetb0(num_classes=train_dataset.num_classes)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoints
    def load_checkpoint_ddp(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        # The model is pretrained on multiple GPUs, so we need to remove the "module" prefix
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
        return model
    
    models = []
    for i in range(args.ensemble_size):
        model_ckpt = os.path.join(args.ckpt_dir, f"{args.model}_{args.dataset}_{i}.pth")
        assert os.path.exists(model_ckpt), f"Checkpoint {model_ckpt} not found"
        model = load_checkpoint_ddp(model, model_ckpt)
        model.eval()
        models.append(copy.deepcopy(model))
    
    # Evaluate
    correct = 0
    total = 0
    for data in tqdm(train_loader):
        inputs, labels = data["image"], data["label"]
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = []
        for model in models:
            model = model.to(device)
            with torch.no_grad():
                outputs.append(model(inputs))
        outputs = torch.stack(outputs)  # [ensemble_size, batch_size, num_classes]
        outputs = torch.mean(outputs, dim=0)  # [batch_size, num_classes]
        
        _, predicted = torch.max(outputs, 1)  # [batch_size]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Train accuracy: {train_acc}")
    
    correct = 0
    total = 0
    for data in tqdm(test_loader):
        inputs, labels = data["image"], data["label"]
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = []
        for model in models:
            model = model.to(device)
            with torch.no_grad():
                outputs.append(model(inputs))
        outputs = torch.stack(outputs)  # [ensemble_size, batch_size, num_classes]
        
        outputs = torch.mean(outputs, dim=0)  # [batch_size, num_classes]
        _, predicted = torch.max(outputs, 1)  # [batch_size]        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Test accuracy: {test_acc}")
    
    # Single model evaluation
    # test only
    print("Single model evaluation")
    single_accs = []
    for i in range(args.ensemble_size):
        model = models[i]
        
        correct = 0
        total = 0
        for data in tqdm(test_loader):
            inputs, labels = data["image"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Model {i} test accuracy: {acc}")
        single_accs.append(acc)
    print(f"Mean test accuracy: {sum(single_accs) / len(single_accs)}")
    
    
if __name__ == "__main__":
    eval()