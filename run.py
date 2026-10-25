"""
Train a CNN on MNIST from the command line.

Usage:
    python run.py                         # defaults: -e 2 -b 1000 -l 0.001
    python run.py -e 5 -b 500 -l 0.0005
"""

import torch

from utilities import get_args, ConvNet, MNISTData, Trainer, plot


def main():
    args = get_args()

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | num_epochs: {args.num_epochs} | "
          f"batch_size: {args.batch_size} | learning_rate: {args.learning_rate}")

    data = MNISTData(batch_size=args.batch_size)
    model = ConvNet()
    print(model)

    trainer = Trainer(model, data, learning_rate=args.learning_rate, device=device)
    model, loss_list, acc_train, acc_test = trainer.train(args.num_epochs)

    # Name the figure after the run's flags so runs with different values don't overwrite each other
    save_path = f"./figure_e{args.num_epochs}_b{args.batch_size}_l{args.learning_rate}.png"
    plot(loss_list, acc_train, acc_test, save_path=save_path)
    print(f"Figure saved to {save_path}")


if __name__ == '__main__':
    main()
