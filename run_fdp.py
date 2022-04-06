import warnings
warnings.simplefilter("ignore")

from model import load_model
from dataset import load_data
from utils import parse_args_fed

import numpy as np

import torch
import torch.nn as nn

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from tqdm import tqdm
from sklearn import metrics

import copy
import time

def aggregate_model_weights(master_model, worker_models, worker_weights, args):
    worker_params = {worker_id: worker_model.state_dict() for worker_id, worker_model in enumerate(worker_models)}
    new_params = copy.deepcopy(worker_params[0])
    new_params = {name.replace('_module.',''): params for name,params in new_params.items()}  # names
    for name in new_params:
        new_params[name] = torch.zeros(new_params[name].shape, device=args.DEVICE)
    for worker_id, params in worker_params.items():
        for name in new_params:
            new_params[name] += params['_module.'+ name] * worker_weights[worker_id]  # averaging

    master_model.load_state_dict(new_params)
    return master_model.state_dict()

def broadcast_model_weights(new_params, worker_models, args):
    for worker in worker_models:
        params = copy.deepcopy(new_params)
        params = {'_module.'+ name: params for name,params in new_params.items()}
        worker.load_state_dict(params)

def train(worker_id, model, criterion, train_loader, optimizer, epoch, privacy_engine, args):
    model.train()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=args.MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(args.DEVICE)
            target = target.to(args.DEVICE)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = metrics.accuracy_score(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(args.DELTA)
                print(
                    f'Worker: {worker_id} \t'
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.DELTA})"
                )

def test(model, test_loader, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(args.DEVICE)
            target = target.to(args.DEVICE)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            predict_all = np.append(predict_all, preds)
            labels = target.detach().cpu().numpy()
            labels_all = np.append(labels_all, labels)
            losses.append(loss.item())


    acc = metrics.accuracy_score(predict_all, labels_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=args.LABEL_NAMES, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, np.mean(losses), report, confusion


def main(args):
    master_model = load_model(args.MODEL_NAME, args.NUM_CLASSES, args.DEVICE, args.USE_OPACUS)
    train_loaders, test_loader, labelNames = load_data(args)
    args.LABEL_NAMES = labelNames
    worker_models, worker_dataloaders, worker_weights, worker_optimizers, worker_privacy_engines = [], [], [], [], []

    for i in range(args.NUM_WORKERS):
        worker_model = copy.deepcopy(master_model)
        worker_model.to(args.DEVICE)
        
        worker_dataloader, worker_weight = train_loaders[i]
        worker_weights.append(worker_weight)
        
        worker_optimizer = getattr(torch.optim, args.OPTIMIZER)(worker_model.parameters(), args.LR)
        
        worker_privacy_engine = PrivacyEngine()
        worker_model, worker_optimizer, worker_dataloader = worker_privacy_engine.make_private_with_epsilon(module=worker_model,
                                                                                        optimizer=worker_optimizer,
                                                                                        data_loader=worker_dataloader,
                                                                                        epochs=args.EPOCHS,
                                                                                        target_epsilon=args.EPSILON,
                                                                                        target_delta=args.DELTA,
                                                                                        max_grad_norm=args.MAX_GRAD_NORM)
        worker_privacy_engines.append(worker_privacy_engine)
        worker_models.append(worker_model)
        worker_dataloaders.append(worker_dataloader)
        worker_optimizers.append(worker_optimizer)

    criterion = getattr(nn, args.CRITERION)()
    for round in range(args.ROUNDS):
        print('='*20 + f"Round {round}" + '='*20)
        for worker_id in range(args.NUM_WORKERS):
            worker_model = worker_models[worker_id]
            worker_optimizer = worker_optimizers[worker_id]
            worker_privacy_engine = worker_privacy_engines[worker_id]
            worker_dataloader = worker_dataloaders[worker_id]
            for epoch in range(args.EPOCHS):
                train(worker_id, worker_model, criterion, worker_dataloader, worker_optimizer, epoch, worker_privacy_engine, args)

        new_params = aggregate_model_weights(master_model, worker_models, worker_weights, args)
        broadcast_model_weights(new_params, worker_models, args)

        test_acc, test_loss, test_report, test_confusion = test(master_model, test_loader, args)
        print('Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        print()

        

if __name__ == '__main__':
    args = parse_args_fed()
    main(args)


