# -*- Python -*-

import os
import time
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
from torch.distributed.algorithms.join import Join
import contextlib


try:
    from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
except ImportError:
    class ApexDistributedDataParallel:
        pass


def identity(x):
    """Identity function."""
    return x


def fmt(v):
    """Generic number format."""
    if isinstance(v, int):
        return "%9d" % v
    elif isinstance(v, float):
        return "%10.4g" % v
    else:
        return str(v)[:20]


class TextLogger:
    def rank(self):
        if torch.distributed.is_initialized():
            return f"[rank:{torch.distributed.get_rank()}] "
        else:
            return ""

    def message(self, *args):
        """Log a message."""
        msg = " ".join([str(x) for x in args])
        print(self.rank(), msg, file=sys.stderr)

    def __bool__(self):
        """Check whether the logger is active."""
        return True

    def params(self, **kw):
        """Log a parameter."""
        print(self.rank(), kw, file=sys.stderr)

    def log(self, prefix="train/", step=None, **kw):
        """Log to a time series."""
        st = "@" + str(step) if isinstance(step, int) else ""
        st = (" " * 10 + st)[-10:]
        print(self.rank(), prefix, st, " ".join([f"{k}:{fmt(v)}" for k, v in kw.items()]), file=sys.stderr)

    def upload(self, name, object, step=None):
        """Upload an object."""
        pass


class TensorboardLogger:
    def __init__(self, log_dir=None, comment=None):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def message(self, *args):
        """Log a message."""
        pass

    def params(self, **kw):
        """Log a parameter."""
        # FIXME -- change to self.hparams(kw) interface
        pass

    def log(self, prefix="train/", step=None, **kw):
        """Log to a time series."""
        for k, v in kw.items():
            self.writer.add(prefix+k, v, step)

    def upload(self, name, object, step=None):
        """Upload an object."""
        pass

class Loggers:
    """A logging class that forwards to other loggers."""
    def __init__(self, loggers=None, enable=True):
        """Initialize."""
        if loggers is None:
            loggers = [TextLogger()]
        self.loggers = loggers if enable else []

    def message(self, *args):
        """Log a message."""
        for logger in self.loggers:
            if logger:
                logger.message(*args)

    def params(self, **kw):
        """Log a parameter."""
        for logger in self.loggers:
            if logger:
                logger.params(**kw)

    def log(self, **kw):
        """Log to a time series."""
        for logger in self.loggers:
            if logger:
                logger.log(**kw)

    def upload(self, name, obj, step=None):
        """Upload an object."""
        for logger in self.loggers:
            if logger:
                logger.upload(name, obj, step=step)


def schedule(epoch):
    """A simple learning rate schedule."""
    return 0.1 ** (epoch // 30)


def train_classifier(
    model,
    loader,
    step=0,
    lossfn=F.cross_entropy,
    optimizer=None,
    scheduler=None,
    logger=None,
    loginterval=10.0,
    amplevel="",
    join=False,
):
    """Train for one epoch. Handles DDP and logging."""
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4,)
    scheduler = scheduler or lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    device = next(model.parameters()).device
    model.train()
    last, last_step = time.time(), 0
    context = Join([model]) if join else contextlib.nullcontext()
    with context:
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = lossfn(output, target)
            if amplevel != "":
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            step += len(images) * world_size
            lr = scheduler.get_last_lr()[-1]
            err = float((output.argmax(1) != target).sum() / float(len(target)))
            now = time.time()
            if logger is not None and now - last > loginterval:
                rate = (step - last_step) / (now - last)
                last, last_step = now, step
                logger.log(
                    prefix="train/", step=step, loss=float(loss), err=err, lr=lr, rate=rate,
                )
    return step


def evaluate_classifier(model, loader, lossfn=F.cross_entropy, logger=None, prefix="val/", step=None):
    if isinstance(model, (DistributedDataParallel, ApexDistributedDataParallel)):
        # for evaluation, we use the model independently
        model = model.module
    device = next(model.parameters()).device
    losses = []
    errs = []
    model.eval()
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(images)
            loss = lossfn(output, target)
            err = float((output.argmax(1) != target).sum() / float(len(target)))
            losses.append(float(loss))
            errs.append(err)
    total, loss, err = len(losses), np.sum(losses), np.sum(errs)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        result = torch.tensor([total, loss, err]).to(device)
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        total, loss, err = result.tolist()
    loss, err = loss / total, err / total
    if logger is not None:
        logger.log(prefix=prefix, step=step, loss=loss, err=err)
    return loss, err
