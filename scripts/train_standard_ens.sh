    # parser.add_argument("--dataset", type=str, default="cifar10")
    # parser.add_argument("--data_dir", type=str, default="data")
    
    # # Training settings for the model
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--optimizer", type=str, default="sgd")
    # parser.add_argument("--model", type=str, default="resnet18")
    # parser.add_argument("--model_path", type=str, default=None)
    # parser.add_argument("--scheduler", type=str, default=None)
    
    # # Ensemble settings [train]
    # parser.add_argument("--ensemble_size", type=int, default=5)
    # parser.add_argument("--ensemble_type", type=str, default="noisy")  # "noisy" or "standard"
    # parser.add_argument("--perturbation", type=str, default="uniform")  # "uniform" or "normal" or "none"
    # parser.add_argument("--perturbation_type", type=str, default="additive")
    # parser.add_argument("--reset_lr_scheduler", action="store_true")
    # parser.add_argument("--perturbation_strength", type=float, default=0.1)
    # parser.add_argument("--perturbation_ratio", type=float, default=0.1)
    # parser.add_argument("--perturbation_mean", type=float, default=0.0)
    # # parser.add_argument("--perturbation_scheduler", type=str, default=None)
    # # parser.add_argument("--perturbation_epochs", type=int, default=10)
    # # parser.add_argument("--perturbation_decay", type=float, default=0.1)
    # parser.add_argument("--aux_epochs", type=int, default=10)
    
    # # Ensemble settings [test]
    # parser.add_argument("--test", action="store_true")
    # parser.add_argument("--kl_divergence", action="store_true")
    # parser.add_argument("--disagreement", action="store_true")
    # parser.add_argument("--ensemble_path_list", type=str, nargs="+", default=None)
    # parser.add_argument("--ensemble_aggregation", type=str, default="mean")
    
    # # Additional settings
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # parser.add_argument("--log_interval", type=int, default=10)
    # parser.add_argument("--save_interval", type=int, default=1)
    # parser.add_argument("--save_dir", type=str, default="models")

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12357
python src/train.py \
    --dataset cifar100 \
    --data_dir data \
    --batch_size 512 \
    --num_workers 1 \
    --epochs 100 \
    --lr 0.1 \
    --optimizer sgd \
    --model resnet18 \
    --model_path None \
    --scheduler cosine \
    --ensemble_size 1 \
    --ensemble_type standard \
    --device cuda \
    --log_interval 10 \
    --save_interval 100 \
    --save_dir models/aux/sgd # models/standard_m10_resnet18_cifar100_ep100_sgd