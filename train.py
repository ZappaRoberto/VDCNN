import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import VDCNN
from utils import (
    get_loaders,
    save_checkpoint,
    load_checkpoint,
    check_accuracy,
    save_plot)

# Hyper parameters
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
MAX_LENGTH = 1024
NUM_EPOCHS = 2  # 15
PATIENCE = None
NUM_WORKERS = 12
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"


def train_fn(epoch, loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch}", refresh=True)

    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        # forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, target)

        # backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = prediction.max(dim=1)

        # accuracy
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    loop.close()
    train_loss = running_loss / len(loop)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy


def controller():
    pass


def main():
    torch.backends.cudnn.benchmark = True
    model = VDCNN(depth=9, n_classes=10).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_loaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE, MAX_LENGTH, NUM_WORKERS, PIN_MEMORY)
    train_l, train_a, test_l, test_a = [], [], [], []

    accuracy = 0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_fn(epoch, train_loader, model, optimizer, loss_fn, scaler)
        # check accuracy
        test_loss, test_accuracy = check_accuracy(test_loader, model, loss_fn, device=DEVICE)
        train_l.append(train_loss)
        train_a.append(train_accuracy)
        test_l.append(test_loss)
        test_a.append(test_accuracy)
        # save model
        if test_accuracy > accuracy:
            accuracy = test_accuracy
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
    save_plot(train_l, train_a, test_l, test_a)
    sys.exit()


if __name__ == "__main__":
    main()
