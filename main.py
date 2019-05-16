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
from nvidia.dali.plugin import pytorch
from apex import amp
from apex import parallel
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
args = parser.parse_args()

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
    global_rank = distributed.get_rank()
    device_count = torch.cuda.device_count()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    print(f'Enabled distributed training. (global_rank: {global_rank}/{world_size}, local_rank: {local_rank}/{device_count})')

    torch.manual_seed(0)
    model = models.resnet50().cuda()

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level)
    model = parallel.DistributedDataParallel(model, delay_allreduce=True)

    last_epoch = -1
    if args.checkpoint:
        checkpoint = Dict(torch.load(args.checkpoint), map_location=lambda storage, location: storage.cuda(local_rank))
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        last_epoch = checkpoint.last_epoch

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.lr_milestones,
        gamma=config.lr_gamma,
        last_epoch=last_epoch
    )

    if args.training:

        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)

        # NOTE: When partition for distributed training executed?
        # NOTE: Should random seed be the same in the same node?
        train_pipeline = TrainPipeline(
            root=config.train_root,
            batch_size=config.batch_size,
            num_threads=config.num_workers,
            device_id=local_rank,
            num_shards=device_count,
            shard_id=local_rank,
            image_size=224
        )
        train_pipeline.build()

        # NOTE: What's `epoch_size`?
        # NOTE: Is that len(dataset) ?
        train_data_loader = pytorch.DALIClassificationIterator(
            pipelines=train_pipeline,
            size=list(train_pipeline.epoch_size().values())[0] / world_size
        )

        val_pipeline = ValPipeline(
            root=config.val_root,
            batch_size=config.batch_size,
            num_threads=config.num_workers,
            device_id=local_rank,
            num_shards=device_count,
            shard_id=local_rank,
            image_size=224
        )
        val_pipeline.build()

        val_data_loader = pytorch.DALIClassificationIterator(
            pipelines=val_pipeline,
            size=list(val_pipeline.epoch_size().values())[0] / world_size
        )

        summary_writer = SummaryWriter(config.event_directory)

        for epoch in range(last_epoch + 1, config.num_epochs):

            model.train()

            scheduler.step()

            for step, data in enumerate(train_data_loader):

                images = data[0]["data"]
                labels = data[0]["label"]

                images = images.cuda()
                labels = labels.cuda()
                labels = labels.squeeze().long()

                logits = model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                if global_rank == 0:

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

                for step, data in enumerate(val_data_loader):

                    images = data[0]["data"]
                    labels = data[0]["label"]

                    images = images.cuda()
                    labels = labels.cuda()
                    labels = labels.squeeze().long()

                    logits = model(images)
                    loss = criterion(logits, labels) / world_size
                    distributed.all_reduce(loss)

                    predictions = logits.topk(1)[1].squeeze()
                    correct = torch.sum(predictions == labels)

                    total_loss += loss
                    total_correct += correct

                loss = total_loss / len(val_data_loader)
                accuracy = total_correct / len(val_data_loader)

            if global_rank == 0:

                summary_writer.add_scalars(
                    main_tag='validation',
                    tag_scalar_dict=dict(loss=loss, accuracy=accuracy),
                    global_step=len(train_data_loader) * epoch + step
                )

                print(f'[validation] epoch: {epoch} loss: {loss} accuracy: {accuracy}')


if __name__ == '__main__':
    main()
