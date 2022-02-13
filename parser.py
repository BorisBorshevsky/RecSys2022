import argparse


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Auto Rec')
    parser.add_argument('--hidden_neuron', type=int, default=500)
    parser.add_argument('--lambda_value', type=float, default=1)

    parser.add_argument('--train_epoch', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp', 'Adagrad'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

    parser.add_argument('--display_step', type=int, default=1)

    return parser


