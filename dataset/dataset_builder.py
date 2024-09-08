from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .mnist import MNISTDataset
from torchvision import transforms as tfms
from torchvision.transforms import Compose

def build_dataset(args, train):
    if train:
        if args.model == "resnet18":
            if args.dataset == "cifar10":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            elif args.dataset == "cifar100":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    tfms.RandomHorizontalFlip(p=0.5),
                    tfms.RandomRotation(degrees=[-15, 15]),
                ])
        elif args.model == "vgg16":
            if args.dataset == "cifar10":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                    tfms.RandomHorizontalFlip(p=0.5),
                    tfms.RandomRotation(degrees=[-15,15])
                ])
            elif args.dataset == "cifar100":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                    tfms.RandomHorizontalFlip(p=0.5),
                    tfms.RandomRotation(degrees=[-15,15])
                ])
        elif args.model == "efficientnet_b0":
            if args.dataset == "cifar10":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Resize(size=(224, 224), antialias=True),
                    tfms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                    tfms.RandomHorizontalFlip(p=0.5),
                    tfms.RandomRotation(degrees=[-15,15])
                ])
            elif args.dataset == "cifar100":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Resize(size=(224, 224), antialias=True),
                    tfms.Normalize(
                        [0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]
                    ),
                    tfms.RandomHorizontalFlip(p=0.5),
                    tfms.RandomRotation(degrees=[-15,15])
                ])
        else:
            raise ValueError("Invalid dataset")
    else:
        if args.model == "resnet18":
            if args.dataset == "cifar10":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            elif args.dataset == "cifar100":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        elif args.model == "vgg16":
            if args.dataset == "cifar10":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                ])
            elif args.dataset == "cifar100":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                ])
        elif args.model == "efficientnet_b0":
            if args.dataset == "cifar10":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Resize(size=(224, 224), antialias=True),
                    tfms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    ),
                ])
            elif args.dataset == "cifar100":
                transforms = Compose([
                    tfms.ToTensor(),
                    tfms.Resize(size=(224, 224), antialias=True),
                    tfms.Normalize(
                        [0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]
                    )
                ])
        else:
            raise ValueError("Invalid dataset")
        
    if args.dataset == "mnist":
        return MNISTDataset(root=args.data_dir, transform=transforms, download=True, train=train)   
    elif args.dataset == "cifar10":
        return CIFAR10Dataset(root=args.data_dir, transform=transforms, download=True, train=train)
    elif args.dataset == "cifar100":
        return CIFAR100Dataset(root=args.data_dir, transform=transforms, download=True, train=train)
    