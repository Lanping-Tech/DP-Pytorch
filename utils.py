import argparse
import torch

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 30:  # warm-up
        lr = lr * float(epoch + 1) / 30
    else:
        lr = lr * (0.2 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--MAX_GRAD_NORM', default=1.2, type=float, help='The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.')
    parser.add_argument('--EPSILON', default=50, type=float, help='The amount of noise sampled and added to the average of the gradients in a batch.')
    parser.add_argument('--DELTA', default=1e-5, type=float, help='The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. It is set to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.')

    # Data
    parser.add_argument('--DATA_NAME', default='CIFAR10', type=str, help='The name of the dataset.')
    parser.add_argument('--BATCH_SIZE', default=512, type=int, help='The batch size of the dataset.')
    parser.add_argument('--MAX_PHYSICAL_BATCH_SIZE', default=128, type=int, help='The maximum physical batch size of the dataset.')
    # To balance peak memory requirement, which is proportional to batch_size^2, and training performance, we will be using BatchMemoryManager. It separates the logical batch size (which defines how often the model is updated and how much DP noise is added), and a physical batch size (which defines how many samples do we process at a time).
    # With BatchMemoryManager you will create your DataLoader with a logical batch size, and then provide maximum physical batch size to the memory manager.

    # Model
    parser.add_argument('--MODEL_NAME', default='resnet18', type=str, help='The name of the model.')
    parser.add_argument('--NUM_CLASSES', default=10, type=int, help='The number of classes of the dataset.')
    parser.add_argument('--DEVICE', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='The device to use for training and testing.')

    # Training
    parser.add_argument('--EPOCHS', default=5, type=int, help='The number of epochs to train the worker model.')
    parser.add_argument('--LR', default=1e-3, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--OPTIMIZER', default='Adam', type=str, help='The optimizer to use.')
    parser.add_argument('--CRITERION', default='CrossEntropyLoss', type=str, help='The loss function to use.')
    parser.add_argument('--USE_OPACUS', default=True, type=bool, help='Whether to use opacus or not.')

    return parser.parse_args()

def parse_args_fed():
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--MAX_GRAD_NORM', default=1.2, type=float, help='The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.')
    parser.add_argument('--EPSILON', default=50, type=float, help='The amount of noise sampled and added to the average of the gradients in a batch.')
    parser.add_argument('--DELTA', default=1e-5, type=float, help='The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. It is set to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.')
    parser.add_argument('--NUM_WORKERS', default=2, type=float, help='The number of workers to use for training.')

    # Data
    parser.add_argument('--DATA_NAME', default='FedCIFAR10', type=str, help='The name of the dataset.')
    parser.add_argument('--BATCH_SIZE', default=128, type=int, help='The batch size of the dataset.')
    parser.add_argument('--MAX_PHYSICAL_BATCH_SIZE', default=64, type=int, help='The maximum physical batch size of the dataset.')
    # To balance peak memory requirement, which is proportional to batch_size^2, and training performance, we will be using BatchMemoryManager. It separates the logical batch size (which defines how often the model is updated and how much DP noise is added), and a physical batch size (which defines how many samples do we process at a time).
    # With BatchMemoryManager you will create your DataLoader with a logical batch size, and then provide maximum physical batch size to the memory manager.

    # Model
    parser.add_argument('--MODEL_NAME', default='resnet18', type=str, help='The name of the model.')
    parser.add_argument('--NUM_CLASSES', default=10, type=int, help='The number of classes of the dataset.')
    parser.add_argument('--DEVICE', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='The device to use for training and testing.')

    # Training
    parser.add_argument('--EPOCHS', default=1, type=int, help='The number of epochs to train the model.')
    parser.add_argument('--ROUNDS', default=10, type=int, help='The number of rounds to train the worker model.')
    parser.add_argument('--LR', default=1e-3, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--OPTIMIZER', default='Adam', type=str, help='The optimizer to use.')
    parser.add_argument('--CRITERION', default='CrossEntropyLoss', type=str, help='The loss function to use.')
    parser.add_argument('--USE_OPACUS', default=True, type=bool, help='Whether to use opacus or not.')

    return parser.parse_args()

    

def parse_args_fed_laplace():
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--MAX_GRAD_NORM', default=1.2, type=float, help='The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.')
    parser.add_argument('--EPSILON', default=50, type=float, help='The amount of noise sampled and added to the average of the gradients in a batch.')
    parser.add_argument('--SIGMA', default=0.1, type=float, help='Sigma of LabelPrivacy')
    parser.add_argument('--DELTA', default=1e-5, type=float, help='The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. It is set to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.')
    parser.add_argument('--POST_PROCESS', default='mapwithprior', type=str, help='The post process of the dataset.')
    parser.add_argument('--MECHANISM', default='Laplace', type=str, help='The mechanism of the dataset.')
    parser.add_argument('--NOISE_ONLY_ONCE', default=True, type=bool, help='Whether to add noise only once or not.')
    parser.add_argument('--NUM_WORKERS', default=2, type=float, help='The number of workers to use for training.')

    # Data
    parser.add_argument('--CANARY', default=0, type=int, help='The canary of the dataset.')
    parser.add_argument('--DATA_NAME', default='FedCIFAR10LAPLACE', type=str, help='The name of the dataset.')
    parser.add_argument('--BATCH_SIZE', default=128, type=int, help='The batch size of the dataset.')
    parser.add_argument('--MAX_PHYSICAL_BATCH_SIZE', default=64, type=int, help='The maximum physical batch size of the dataset.')
    # To balance peak memory requirement, which is proportional to batch_size^2, and training performance, we will be using BatchMemoryManager. It separates the logical batch size (which defines how often the model is updated and how much DP noise is added), and a physical batch size (which defines how many samples do we process at a time).
    # With BatchMemoryManager you will create your DataLoader with a logical batch size, and then provide maximum physical batch size to the memory manager.

    # Model
    parser.add_argument('--MODEL_NAME', default='resnet18', type=str, help='The name of the model.')
    parser.add_argument('--NUM_CLASSES', default=10, type=int, help='The number of classes of the dataset.')
    parser.add_argument('--DEVICE', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='The device to use for training and testing.')

    # Training
    parser.add_argument('--EPOCHS', default=1, type=int, help='The number of epochs to train the model.')
    parser.add_argument('--ROUNDS', default=10, type=int, help='The number of rounds to train the worker model.')
    parser.add_argument('--LR', default=1e-3, type=float, help='The learning rate of the optimizer.')
    parser.add_argument('--OPTIMIZER', default='Adam', type=str, help='The optimizer to use.')
    parser.add_argument('--USE_OPACUS', default=True, type=bool, help='Whether to use opacus or not.')
    parser.add_argument('--SEED', default=2022, type=int, help='The random seed.')

    return parser.parse_args()