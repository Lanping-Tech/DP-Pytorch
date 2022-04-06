import warnings
warnings.simplefilter("ignore")

from model import load_model
from dataset import load_data
from utils import parse_args

import numpy as np

import torch
import torch.nn as nn

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from tqdm import tqdm
from sklearn import metrics

def train(model, criterion, train_loader, optimizer, epoch, privacy_engine, args):
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
                    f"\tTrain Epoch: {epoch} \t"
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
    model = load_model(args.MODEL_NAME, args.NUM_CLASSES, args.DEVICE, args.USE_OPACUS)
    criterion = getattr(nn, args.CRITERION)()
    optimizer = getattr(torch.optim, args.OPTIMIZER)(model.parameters(), args.LR)
    train_loader, test_loader, labelNames = load_data(args)
    args.LABEL_NAMES = labelNames

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module=model,
                                                                              optimizer=optimizer,
                                                                              data_loader=train_loader,
                                                                              epochs=args.EPOCHS,
                                                                              target_epsilon=args.EPSILON,
                                                                              target_delta=args.DELTA,
                                                                              max_grad_norm=args.MAX_GRAD_NORM)
    
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.MAX_GRAD_NORM}")

    

    for epoch in tqdm(range(args.EPOCHS), desc="Epoch", unit="epoch"):
        train(model, criterion, train_loader, optimizer, epoch, privacy_engine, args)
    

    test_acc, test_loss, test_report, test_confusion = test(model, test_loader, args)
    print('Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

        

if __name__ == '__main__':
    args = parse_args()
    main(args)


