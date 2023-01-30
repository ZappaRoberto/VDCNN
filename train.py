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
NUM_WORKERS = ''
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

    # forward
    with torch.cuda.amp.autocast():
        pass


def main():
    model = VDCNN(depth=9, n_classes=10).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()  # control if after a softmax is correct to use crossentropyloss
    train_loader, test_loader = get_loaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS, MAX_LENGTH, PIN_MEMORY)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)
        pass


if __name__ == "__main__":
    main()
