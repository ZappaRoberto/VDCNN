import torch
from model import VDCNN
from utils import load_checkpoint
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TASK = 'classification'  # 'sentimental'
WEIGHT_DIR = "result/something/checkpoint.pth.tar"


def load_model():
    model = VDCNN(depth=9, n_classes=10, want_shortcut=False, pool_type='resnet').to(DEVICE)
    load_checkpoint(torch.load(WEIGHT_DIR), model)
    model.eval()
    return model


def input_data():
    vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%^&*~‘+=<>()[]{} """)
    max_length = 1024
    string = input()
    string = ' '.join(string.split())
    string = string[:max_length]
    tokenizer = []
    for char in list(string):
        if char in vocabulary:
            tokenizer.append(vocabulary.index(char) + 1)  # 0 is for padding
        else:
            tokenizer.append(len(vocabulary) + 1)  # 68 is for unknown character
    if len(tokenizer) < max_length:
        tokenizer += [0] * (max_length - len(tokenizer))
    data = torch.from_numpy(np.array(tokenizer, dtype=np.int64)).to(DEVICE)
    return data


def prediction(model, data):
    classification = ['Society & Culture', 'Science & Mathematics', 'Health',
                      'Education & Reference', 'Computers & Internet',
                      'Sports', 'Business & Finance', 'Entertainment & Music',
                      'Family & Relationships', 'Politics & Government']
    response = model(data)
    response = response.max(dim=1).item()
    if TASK == 'classification':
        response = classification[response]
    else:
        response += 1
    return response


def main():
    model = load_model()
    data = input_data()
    response = prediction(model, data)
    print(response)


if __name__ == "__main__":
    main()
