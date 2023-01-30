from torch.utils.data import Dataset
import pandas as pd
import cupy as cy

# 67 + padding -> (0) and unknown token -> (69)
VOCABULARY = list("""£abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%^&*~‘+=<>()[]{} """)


class YahooDataset(Dataset):
    def __init__(self, path, max_length=1024):
        df = pd.read_csv(path,
                         delimiter=',',
                         names=['label', 'question_title', 'question_content', 'best_answer']).iloc[:, :-1]

        df = df.fillna('')
        df = df.astype(str)
        df['question_title'] = df['question_title'].str.lower()
        df['question_content'] = df['question_content'].str.lower()

        self.text, self.label = [], []

        for row in df.itertuples():
            self.label.append(row.label)
            string = row.question_title + ' ' + row.question_content
            string = string.replace(r'\n', ' ')
            string = string.replace('  ', ' ')
            tokenizer = []
            for char in list(string):
                if char in VOCABULARY:
                    tokenizer.append(VOCABULARY.index(char) + 1)  # 0 is for padding
                else:
                    tokenizer.append(len(VOCABULARY) + 2)  # 69 is for unknown character
            if len(tokenizer) > max_length:
                tokenizer = tokenizer[:max_length]
            elif len(tokenizer) < max_length:
                tokenizer += [0] * (max_length - len(tokenizer))
            self.text.append(tokenizer)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return cy.array(self.text[index], dtype=cy.int64), self.label[index]


if __name__ == "__main__":
    YahooDataset("dataset/test.csv")

