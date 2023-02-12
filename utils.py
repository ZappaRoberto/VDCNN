import torch
from torch.utils.data import DataLoader
from dataset import YahooDataset, AmazonDataset
import matplotlib.pyplot as plt


def save_checkpoint(state, filename="result/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir, test_dir, batch_size, max_length, num_workers, pin_memory):

    train_ds = YahooDataset(
        path=train_dir,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = YahooDataset(
        path=test_dir,
        max_length=max_length
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader


def check_accuracy(loader, model, loss_fn, scheduler, device):
    running_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(data)
            loss = loss_fn(prediction, target)
            running_loss += loss.item()
            _, predicted = prediction.max(dim=1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = correct / total * 100
    loss = running_loss / len(loader)
    scheduler.step(loss)

    print(f"Got on test set Accuracy: {accuracy:.3f} and Loss: {loss:.3f}")
    model.train()
    return loss, accuracy


def save_plot(train_l, train_a, test_l, test_a):
    plt.plot(train_a, '-')
    plt.plot(test_a, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.savefig('result/accuracy')
    plt.close()

    plt.plot(train_l, '-')
    plt.plot(test_l, '-')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig('result/losses')
    plt.close()
