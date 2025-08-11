import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self, dna_emb_size=768, image_emb_size=1000, num_classes=2):
        super(Decoder, self).__init__()
        self.dna_emb_size = dna_emb_size
        self.image_emb_size = image_emb_size

        self.fc1 = nn.Linear(dna_emb_size + image_emb_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
