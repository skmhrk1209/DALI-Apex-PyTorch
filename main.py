import torch
from torch import nn
from torch import distributed
from torch import optim
from torch import utils
from torch import cuda
from torch import backends
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from tensorboardX import SummaryWriter
from nvidia import dali
from apex import parallel
from apex import fp16_utils
from pipelines import TrainPipeline
from pipelines import ValPipeline
import argparse
import json
import os

parser = argparse.ArgumentParser(description='ResNet50 training on Imagenet')
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training', action='store_true')
parser.add_argument('--evaluation', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument("--local_rank", type=int)
args, unkown = parser.parse_known_args()

backends.cudnn.benchmark = True


class Dict(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


def main():

    with open(args.config) as file:
        config = Dict(json.load(file))

    distributed.init_process_group(backend='nccl')
    world_size = distributed.get_world_size()
    rank = distributed.get_rank()
    num_gpu = torch.cuda.device_count()
    gpu = rank % num_gpu
    torch.cuda.set_device(gpu)
    print(f'Enabled distributed training. (rank {rank}/{world_size})')

    torch.manual_seed(0)
    model = models.resnet50().cuda()
    model = fp16_utils.network_to_half(model)
    model = parallel.DistributedDataParallel(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    optimizer = fp16_utils.FP16_Optimizer(
        init_optimizer=optimizer,
        static_loss_scale=config.static_loss_scale,
        dynamic_loss_scale=config.dynamic_loss_scale
    )

    last_epoch = -1
    if args.checkpoint:
        checkpoint = Dict(torch.load(args.checkpoint), map_location=lambda storage, location: storage.cuda(gpu))
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        last_epoch = checkpoint.last_epoch

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.lr_milestones,
        gamma=config.lr_gamma,
        last_epoch=last_epoch
    )

    if args.training:

        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)

        train_pipeline = TrainPipeline(
            root=config.train_root,
            batch_size=config.batch_size,
            num_threads=config.num_workers,
            device_id=0,
            num_shards=num_gpu,
            shard_id=gpu,
            image_size=224
        )
        train_pipeline.build()

        train_data_loader = dali.plugin.pytorch.DALIClassificationIterator(train_pipeline)

        val_pipeline = ValPipeline(
            root=config.val_root,
            batch_size=config.batch_size,
            num_threads=config.num_workers,
            device_id=0,
            num_shards=num_gpu,
            shard_id=gpu,
            image_size=224
        )
        val_pipeline.build()

        val_data_loader = dali.plugin.pytorch.DALIClassificationIterator(val_pipeline)

        summary_writer = SummaryWriter(config.event_directory)

        for epoch in range(last_epoch + 1, config.num_epochs):

            model.train()

            scheduler.step()

            for step, (images, labels) in enumerate(train_data_loader):

                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()

                if rank == 0:

                    summary_writer.add_scalars(
                        main_tag='training',
                        tag_scalar_dict=dict(loss=loss),
                        global_step=len(train_data_loader) * epoch + step
                    )

                    print(f'[training] epoch: {epoch} step: {step} loss: {loss}')

            torch.save(dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                last_epoch=epoch
            ), f'{config.checkpoint_directory}/epoch_{epoch}')

            model.eval()

            total_loss = 0
            total_correct = 0

            with torch.no_grad():

                for images, labels in val_data_loader:

                    images = images.cuda()
                    labels = labels.cuda()
                    logits = model(images)

                    loss = criterion(logits, labels) / world_size
                    distributed.all_reduce(loss)

                    predictions = logits.topk(1)[1].squeeze()
                    correct = torch.sum(predictions == labels)

                    total_loss += loss
                    total_correct += correct

                loss = total_loss / len(val_data_loader)
                accuracy = total_correct / len(val_data_loader)

            if rank == 0:

                summary_writer.add_scalars(
                    main_tag='validation',
                    tag_scalar_dict=dict(loss=loss, accuracy=accuracy),
                    global_step=len(train_data_loader) * epoch + step
                )

                print(f'[validation] epoch: {epoch} loss: {loss} accuracy: {accuracy}')


if __name__ == '__main__':
    main()
