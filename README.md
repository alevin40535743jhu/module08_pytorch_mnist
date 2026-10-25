# module08_pytorch_mnist

Object-oriented PyTorch program that trains a Convolutional Neural Network (CNN) to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (60,000 training and 10,000 test grayscale images, 28×28 pixels, 10 classes). Training hyperparameters are passed from the command line with `argparse`.

## Files

| File | Purpose |
|---|---|
| `utilities.py` | Argument parser (`get_args`), CNN model (`ConvNet`), data loading (`MNISTData`), training/evaluation (`Trainer`), and the `plot` function |
| `run.py` | Entry point: parses the flags, trains the model, and saves the loss/accuracy figure |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes `__pycache__/` and the downloaded `data/` folder |

## Setup

```sh
pip install -r requirements.txt
```

On a CPU-only machine (e.g., a GitHub Codespace), installing the CPU build of PyTorch first avoids downloading the much larger CUDA build:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage

| Short flag | Long flag | Description | Default |
|---|---|---|---|
| `-e` | `--num_epochs` | Number of training epochs | `2` |
| `-b` | `--batch_size` | Samples per batch | `1000` |
| `-l` | `--learning_rate` | Adam learning rate | `0.001` |

Train with the default values:

```sh
python run.py
```

Train with custom values:

```sh
python run.py -e 5 -b 500 -l 0.0005
```

Show help:

```sh
python run.py -h
```

MNIST is downloaded automatically to `./data` on the first run. A GPU is used if available; otherwise training runs on the CPU.

## Output

Each epoch prints the average training loss and the train/test accuracy. At the end, a two-panel figure (training loss and train/test accuracy) is saved as `figure_e<epochs>_b<batch>_l<lr>.png`, e.g. `figure_e2_b1000_l0.001.png` for the defaults.

## Model

Two convolutional blocks (Conv2d 5×5 → BatchNorm → ReLU → MaxPool 2×2; 32 then 64 channels), followed by fully connected layers 3136 → 1000 → 10. Trained with cross-entropy loss and the Adam optimizer.
