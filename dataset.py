from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# 67 + padding -> (0) and unknown token -> (69)
VOCABULARY = list("""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%^&*~‘+=<>()[]{} """)


class YahooDataset(Dataset):
    def __init__(self, path, max_length=1024):
        df = pd.read_csv(path,
                         delimiter=',',
                         names=['label', 'question_title', 'question_content', 'best_answer']).iloc[:, :-1]

        df = df.fillna('')
        df = df.astype(str)
        df['label'] = df['label'].astype(int)
        df['question_title'] = df['question_title'].str.lower()
        df['question_content'] = df['question_content'].str.lower()

        self.max_length = max_length
        self.text, self.label = [], []

        for row in df.itertuples():
            self.label.append(row.label - 1)  # from 0 to 9
            string = " ".join([row.question_title, row.question_content])
            string = string.replace(r'\n', ' ')
            string = string[:self.max_length]
            string = ' '.join(string.split())
            self.text.append(string)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        tokenizer = []
        for char in list(self.text[index]):
            if char in VOCABULARY:
                tokenizer.append(VOCABULARY.index(char) + 1)  # 0 is for padding
            else:
                tokenizer.append(len(VOCABULARY) + 1)  # 68 is for unknown character
        if len(tokenizer) < self.max_length:
            tokenizer += [0] * (self.max_length - len(tokenizer))
        return np.array(tokenizer, dtype=np.int64), np.array(self.label[index], dtype=np.int64)


if __name__ == "__main__":
    YahooDataset("dataset/test.csv")
