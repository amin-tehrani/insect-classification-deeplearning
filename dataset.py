import torch
from torch.utils.data import Dataset
import numpy as np

class MultiModalDataset:
    def __init__(self, dna_strings, images, labels, dna_str_len_mapping, species2genus, genus_species, img_processor, dna_tokenizer, add_genus=True, max_length=1600):
        self.images = images
        self.dna_strings = dna_strings
        self.labels = labels
        self.img_processor = img_processor
        self.dna_tokenizer = dna_tokenizer
        self.dna_str_len_mapping = dna_str_len_mapping
        self.species2genus = species2genus - 1
        self.max_length = max_length
        self.genus_species = genus_species
        self.add_genus = add_genus

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # ===== Image Processing =====
        # np.transpose(self.images[idx],(0, 2, 3, 1))
        if self.images[idx].ndim == 3:
            self.images[idx] = np.expand_dims(self.images[idx], axis=0)
        image = np.transpose(self.images[idx],(1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        

        image_inputs = self.img_processor(images=image, return_tensors="pt")

        # ===== DNA Processing =====
        dna_sequence = self.dna_strings[idx].strip()
        dna_len_token = self.dna_str_len_mapping[len(dna_sequence)]

        dna_inputs = self.dna_tokenizer(
            dna_sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # ===== Label & Genus =====
        label = torch.tensor(self.labels[idx], dtype=torch.long) - 1
        genus = torch.tensor(self.species2genus[label], dtype=torch.long)

        res = {
            'dna_len_tokens': torch.tensor(dna_len_token, dtype=torch.long),
            'image_inputs': image_inputs,
            'dna_inputs': dna_inputs,
            'labels': label,
        } 
        if self.add_genus:
            res['genus'] = genus
            
        return res

