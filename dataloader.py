import torch
from torch.utils.data import Dataset, DataLoader

class TwitterDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = self.df.text[item]
        label = self.df.label[item]

        encoding = self.tokenizer.encode_plus(
            text, #w/o .item()
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            #padding="longest",    
            pad_to_max_length=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),
        }
    

class TwitterLogits(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        inputs = self.X[idx]
        label = self.y[idx]
        
        return {
            "inputs": torch.from_numpy(inputs),
            "labels": torch.tensor(label, dtype=torch.float),
        }