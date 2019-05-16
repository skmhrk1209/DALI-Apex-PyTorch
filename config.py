{
    "num_workers": 4,
    "num_epochs": 90,
    "batch_size": 128,
    "lr": 1.6,
    "lr_milestones": [
        30,
        60,
        80
    ],
    "lr_gamma": 0.1,
    "weight_decay": 0.0001,
    "static_loss_scale": 1.0,
    "dynamic_loss_scale": False,
    "checkpoint_directory": "checkpoints",
    "event_directory": "events",
    "train_root": "cifar10/train",
    "val_root": "cifar10/val"
}
