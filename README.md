# Imagenet Training

## Setup

This is a small repository showing the use of the webdataset library for training models on the Imagenet dataset.

To start, you need to convert the Imagenet dataset to shards:

```
$ mkdir shards
$ python3 convert-image.py /.../imagenet-data
```

This should generate a large number of shards in the `./shards` subdirectory.



```python
!ls shards | shardsum
```

    imagenet-train-{000000..001281}.tar
    imagenet-val-{000000..000049}.tar


You first need to install a virtual environment:

```
$ ./run-venv
...
$ . ./venv/bin/activate
$ 
```

You can then run `jupyter lab` and explore the notebooks or run command line programs.

You have several ways of running Imagenet training:

- `./run-local` will run single GPU training
- `ray-demo.ipynb` shows how to use Ray for multinode training
- `ssh-demo.ipynb` shows how to run distributed multinode training directly
- `./run-ray` runs multinode training using Ray from the command line

## Data Serving

In order to avoid having to distribute the data, the scripts will start up an nginx-based web server serving the data shards using `docker run`. You need to have `docker` installed for this to work.

## Ray-Based Training

Ray is a convenient Python-based distributed computing platform. It is probably less common for starting up multi-node training jobs, but it is very convenient and it's the easiest of the different systems to package.

For the Ray demos, you need to start up Ray. You can do that from the command line:

```
$ ray start --head
...
$ ssh host1 "cd /some/dir && . ./venv/bin/activate && ray start --address=http://$(hostname -i):6379"
$ ...
```

This creates a compute cluster on a local network of machines. Note that this repository needs to be installed on all machines.

You can check the cluster status with:

```
$ ray status
```

Once the cluster is up and running, all the Ray-based training will run automatically and you can just start up, say, `./run-ray`. It will query the cluster for available GPUs and start training automatically.

## SSH-Based Training

There are a couple of examples of SSH-based startup of the distributed training jobs. This is fairly cumbersome to do manually, but you can obviously automate it for your environment.

The more usual way of performing this kind of training is using a container management system like Kubernetes or NGC, in combination with `torchrun`. The command line program is identical to the command line program using ssh-based training, but the environment variables are set for you automatically.

## Distributed Training Modes

A fundamental problem with multinode distributed training is that it is difficult for multiple nodes to have exactly the same number of training batches, since the number of shards may not be evenly divisible by the number of nodes and workers, and since shards may not all contain the same number of samples.

There are two common ways of dealing with this:

1. resampling: each worker picks one of the shards at random and returns samples from this; this leads to some samples being duplicated during each epoch, but is still statistically sound. This is the default. It works well for large datasets and large numbers of shards.
2. exact epochs with `Join`: the shards are split between all the compute nodes and workers and each worker iterates through its data until it runs out; at the end of training, batches become smaller as some of the GPUs have run out of training data. This ensures that each training sample is presented exactly once during each epoch. It differs from "traditional" single GPU training in that batch size is slightly variable. This mode can be enabled with the `--splitshards` argument.

Please see the code for how these different modes are enabled. The command line interface and data loaders are defined in `imagenet.py`, and the training loop itself is in `utils.py`.


