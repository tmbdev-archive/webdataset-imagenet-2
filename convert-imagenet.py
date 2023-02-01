import os
import sys
from torchvision.datasets import ImageNet
import webdataset as wds
import typer

app = typer.Typer()


@app.command()
def convert(root: str, odir: str = "./shards"):
    assert os.path.isdir(root)
    assert os.path.isdir(os.path.join(root, "train"))
    assert os.path.isdir(os.path.join(root, "val"))
    assert os.path.isdir(odir)

    assert not os.path.exists(os.path.join(odir, "imagenet-train-000000.tar"))
    assert not os.path.exists(os.path.join(odir, "imagenet-val-000000.tar"))

    dataset = ImageNet(root="/fast1tb/imagenet-data", split="val")
    opat = os.path.join(odir, "imagenet-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=1000)
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(i, file=sys.stderr)
        img, label = dataset[i]
        output.write({"__key__": "%08d" % i, "jpg": img, "cls": label})
    output.close()

    dataset = ImageNet(root="/fast1tb/imagenet-data", split="train")
    opat = os.path.join(odir, "imagenet-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=1000)
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(i, file=sys.stderr)
        img, label = dataset[i]
        output.write({"__key__": "%08d" % i, "jpg": img, "cls": label})
    output.close()


if __name__ == "__main__":
    app()
