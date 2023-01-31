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
NUM_EPOCHS = 15
NUM_WORKERS = 2
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"


def train_fn(loader, model, optimizer, loss_fn, scaler, running_loss, correct, total):
    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = prediction.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(loop)
    accu = 100. * correct / total

    # update tqdm loop
    loop.set_postfix(loss=loss.item())
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = VDCNN(depth=9, n_classes=10).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_loaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE, MAX_LENGTH, NUM_WORKERS, PIN_MEMORY)

    for epoch in range(NUM_EPOCHS):
        print('\nEpoch : %d' % epoch)
        running_loss = 0
        correct = 0
        total = 0
        train_fn(train_loader, model, optimizer, loss_fn, scaler, running_loss, correct, total)


if __name__ == "__main__":
    main()
