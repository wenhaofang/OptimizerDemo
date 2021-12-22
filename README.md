## Optimizer Demo

This repository includes some demo optimizers.

Note: The project refers to [动手学深度学习](https://zh.d2l.ai/)

Datasets:

* `dataset1`: MNIST

Models:

* `model1`: MLP

Optimizers:

* `optimizer1`: SGD
* `optimizer2`: SGDM
* `optimizer3`: AdaGrad
* `optimizer4`: RMSProp
* `optimizer5`: Adam

### Unit Test

* for loader

```shell
PYTHONPATH=. python loaders/loader.py
```

* for module

```shell
PYTHONPATH=. python modules/module.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each optimizer:

```shell
# optimizer1: SGD
python main.py \
    --optim_type 1 \
    --learning_rate 0.1 \
    --num_epochs 10
```

```shell
# optimizer2: SGDM
python main.py \
    --optim_type 2 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --num_epochs 10
```

```shell
# optimizer3: AdaGrad
python main.py \
    --optim_type 3 \
    --learning_rate 0.01 \
    --eps 1e-10 \
    --num_epochs 10
```

```shell
# optimizer4: RMSProp
python main.py \
    --optim_type 4 \
    --learning_rate 0.01 \
    --eps 1e-8 \
    --gamma 0.99 \
    --num_epochs 10
```

```shell
# optimizer5: Adam
python main.py \
    --optim_type 5 \
    --learning_rate 0.001 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-8 \
    --num_epochs 10
```
