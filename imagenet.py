# -*- Python -*-

import os
import time
import warnings
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.nn import functional as F
from torch.optim import lr_scheduler
import webdataset as wds
import typer
import numpy as np
from itertools import islice
import braceexpand
import sys
from torch.nn.parallel import DistributedDataParallel
import socket
import pprint

import ray

import utils

torchvision
app = typer.Typer()

default_train = "imagenet-train-{000000..001281}.tar"
default_val = "imagenet-val-{000000..000049}.tar"


def make_model(mname):
    if mname == "trivial":
        model1 = nn.Sequential(nn.Flatten(), nn.Linear(224 * 224 * 3, 1000))
    else:
        import torchvision.models
        model1 = eval(f"torchvision.models.{mname}")()
    return model1


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def make_transform(mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif mode == "val":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


def make_loader(urls, mode="train", batch_size=64, num_workers=4, cache_dir=None, resampled=True):

    training = mode == "train"

    transform = make_transform(mode=mode)

    # repeat dataset infinitely for training mode
    repeat = training

    dataset = (
        wds.WebDataset(
            urls,
            repeat=training,
            cache_dir=cache_dir,
            shardshuffle=1000 if training else False,
            resampled=resampled if training else False,
            handler=wds.ignore_and_continue,
            nodesplitter=None if (training and resampled) else nodesplitter,
        )
        .shuffle(5000 if training else 0)
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls", handler=wds.ignore_and_continue)
        .map_tuple(transform)
        .batched(batch_size, partial=False)
    )

    loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)
    return loader

def print_distributed_info():
    keys = "MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK".split()
    for k in keys:
        print(f"{k}={os.environ.get(k)}")

@app.command()
def train(
    bucket: str = "./shards/",
    shards: str = default_train,
    valshards: str = default_val,
    batchsize: int = 32,
    mname: str = "resnet18",
    nepochs: int = 300,
    workers: int = 4,
    ntrain: int = 1281167,
    nval: int = 50000,
    device: str = "cuda",
    loginterval: float = 10.0,
    cache_dir: str = None,
    backend: str = "nccl",
    onelog: bool = False,
    showopen: bool = False,
    amplevel: str = "",
    apexddp: bool = False,
    verbose: bool = False,
    distributed: str = "",
    splitshards: bool = False,
):

    parameters = dict(locals())

    resampled = not splitshards
    print("*** resampled", resampled)

    if showopen:
        os.environ["GOPEN_VERBOSE"] = "1"

    if bucket == "gs:":
        bucket = "pipe:curl -s -L http://storage.googleapis.com/nvdata-imagenet/{}"
    elif bucket == "ais:":
        os.environ["AIS_ENDPOINT"] = "http://ais.dynalias.net:51080"
        bucket = "pipe:ais get gs://nvdata-imagenet/{} - || true"

    if os.environ.get("WORLD_SIZE") is not None:
        print_distributed_info()
        print("init_process_group...")
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        time.sleep(1)
        print("init_process_group done")
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
        print("device", device)
        world_size = torch.distributed.get_world_size()
        multinode = True
        print(
            f"starting rank={torch.distributed.get_rank()} world_size={world_size} device={device}",
            file=sys.stderr,
        )
        print("barrier...")
        torch.distributed.barrier()
        print(
            f"done rank={torch.distributed.get_rank()} world_size={world_size} device={device}",
            file=sys.stderr,
        )
        print("barrier done")
    else:
        if os.getenv("NGC_ARRAY_SIZE") is not None:
            warnings.warn("NGC_ARRAY_SIZE is set but WORLD_SIZE is not")
        multinode = False
        world_size = 1

    if verbose:
        print("logger")

    if multinode and onelog:
        rank0 = (torch.distributed.get_rank() == 0)
        logger = utils.Loggers(enable=rank0)
    else:
        logger = utils.Loggers(enable=True)
    logger.params(**parameters)


    if verbose:
        print("loader")

    bucket = bucket if "{}" in bucket else bucket + "{}"
    shards = bucket.format(shards)
    valshards = bucket.format(valshards)
    logger.message(f"inputs: {shards}")
    logger.message(f"validation: {valshards}")

    loader = make_loader(shards, mode="train", batch_size=batchsize, num_workers=workers, cache_dir=cache_dir, resampled=resampled)
    nbatches = max(1, ntrain // (batchsize * world_size))
    loader = loader.with_epoch(nbatches)

    valloader = make_loader(
        valshards, mode="val", batch_size=batchsize, num_workers=workers, cache_dir=cache_dir
    )
    inputs, targets = next(iter(loader))
    logger.message(
        f"inputs: {tuple(inputs.shape)} {inputs.dtype} {inputs.device} {float(inputs.min())} {float(inputs.max())}"
    )
    logger.message(f"validation: {tuple(targets.shape)}")

    if verbose:
        print("model")

    model = make_model(mname)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.1)

    if amplevel != "":
        from apex import amp
        logger.message(f"using amp level {amplevel}")
        model, optimizer = amp.initialize(model, optimizer, opt_level=amplevel)

    if multinode:
        if apexddp:
            from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
            logger.message(f"using apex ddp")
            model = ApexDistributedDataParallel(model)
        else:
            model = DistributedDataParallel(model, device_ids=[device])

    if verbose:
        print("starting")

    step = 0
    for epoch in range(nepochs):
        logger.message("===", epoch, "===")
        step = utils.train_classifier(
            model,
            loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            step=step,
            loginterval=loginterval,
            amplevel=amplevel,
            join=resampled,
        )
        utils.evaluate_classifier(model, valloader.slice(nval // batchsize), logger=logger, step=step)
        scheduler.step()


@ray.remote(num_gpus=1)
class TrainingWorker:
    def __init__(self, rank: int, wsize: int, backend: str = "gloo"):
        self.rank = rank
        self.wsize = wsize
        self.backend = backend

    def get_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def setup_distributed(self, master):
        self.master = master
        os.environ.update(master)
        os.environ.update(RANK=str(self.rank), WORLD_SIZE=str(self.wsize), BACKEND=self.backend)
        # torch.distributed.init_process_group(backend=self.backend, init_method="env://")

    def train(self, **kw):
        time.sleep(1)
        print("running", self.rank)
        train(**kw)
        print("done", self.rank)


@app.command()
def raytrain(
    bucket: str = "./shards/",
    shards: str = default_train,
    valshards: str = default_val,
    batchsize: int = 32,
    mname: str = "resnet18",
    nepochs: int = 300,
    workers: int = 4,
    ntrain: int = 1281167,
    nval: int = 50000,
    device: str = "cuda:0",
    loginterval: float = 10.0,
    cache_dir: str = None,
    backend: str = "gloo",
    onelog: bool = False,
    showopen: bool = False,
    amplevel: str = "",
    apexddp: bool = False,
    verbose: bool = False,
    wsize: int = 0,
    splitshards: bool = False,
):
    """Train a model on multiple GPUs using Ray.

    You need to start up a ray cluster first, e.g.:

        ray start --head

    You can add additional nodes as specified in the Ray output.

    By default, this will use as many GPUs as are available on
    the cluster.  You can override this with the --wsize option.
    """

    parameters = dict(locals())
    del parameters["wsize"]

    ray.init()
    if wsize == 0:
        wsize = int(ray.available_resources()["GPU"])
    elif wsize == -1:
        wsize = int(ray.cluster_resources()["GPU"])

    print("world size:", wsize)

    print("starting workers")
    workers = [TrainingWorker.remote(i, wsize) for i in range(wsize)]

    print("getting master")
    addr = ray.get(workers[0].get_ip.remote())
    master = dict(MASTER_ADDR=addr, MASTER_PORT="29500")
    print("master", master)

    results = [w.setup_distributed.remote(master) for w in workers]
    ray.get(results)

    print("running")
    results = [w.train.remote(**parameters) for w in workers]
    ray.get(results)
    print("done")

if __name__ == "__main__":
    app()
