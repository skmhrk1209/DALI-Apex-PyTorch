import torch
from nvidia import dali


class TrainPipeline(dali.pipeline.Pipeline):

    def __init__(self, root, batch_size, num_threads, device_id, num_shards, shard_id, image_size):

        super().__init__(batch_size, num_threads, device_id, seed=device_id)

        self.reader = dali.ops.FileReader(
            file_root=root,
            num_shards=num_shards,
            shard_id=shard_id,
            random_shuffle=True
        )

        self.decoder = dali.ops.nvJPEGDecoderRandomCrop(
            device="mixed",
            random_area=[0.1, 1.0],
            random_aspect_ratio=[0.8, 1.25]
        )

        self.resize = dali.ops.Resize(
            device="gpu",
            resize_x=image_size,
            resize_y=image_size
        )

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=image_size,
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
        )

        self.coin = dali.ops.CoinFlip()

    def define_graph(self):
        images, labels = self.reader()
        images = self.decoder(images)
        images = self.resize(images)
        images = self.normalize(images, mirror=self.coin())
        return images, labels


class ValPipeline(dali.pipeline.Pipeline):

    def __init__(self, root, batch_size, num_threads, device_id, num_shards, shard_id, image_size):

        super().__init__(batch_size, num_threads, device_id, seed=device_id)

        self.reader = dali.ops.FileReader(
            file_root=root,
            num_shards=num_shards,
            shard_id=shard_id,
            random_shuffle=False
        )

        self.decoder = dali.ops.nvJPEGDecoder(
            device="mixed"
        )

        self.resize = dali.ops.Resize(
            device="gpu",
            resize_shorter=image_size
        )

        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            crop=image_size,
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            std=(0.229 * 255, 0.224 * 255, 0.225 * 255)
        )

    def define_graph(self):
        images, labels = self.reader()
        images = self.decoder(images)
        images = self.resize(images)
        images = self.normalize(images)
        return images, labels
