This repo is an implementation of DrIT

## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:

```sh
conda env create -f setup/conda.yaml
conda activate dmcgb
sh setup/install_envs.sh
```

## Datasets
Part of this repository relies on external datasets. SODA uses the [Places](http://places2.csail.mit.edu/download.html) dataset for data augmentation, which can be downloaded by running

```sh
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

Distracting Control Suite uses the [DAVIS](https://davischallenge.org/davis2017/code.html) dataset for video backgrounds, which can be downloaded by running

```sh
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

##  To run DrIT

```shell
cd path/to/DrIT
python src/train.py   (DrE if args.if_DrE = Ture)
```


##  Use tensorboard to visualize training and testing
```sh
tensorboard --logdir=logs/path/to/tb
```

