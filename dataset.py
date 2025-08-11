import torch
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, dna_strings, labels, tokenizer, max_len=512):
        self.dna_strings = dna_strings
        self.labels = labels.squeeze()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dna_strings)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.dna_strings[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    

