import torch
from torch.utils.data import Dataset
import numpy as np

def _dna_str_len(s):
    return len(s.strip())

class MultiModalDataset:
    
    def __init__(self, dna_strings, images, labels, dna_str_len_mapping, species2genus, genus_species, img_processor, dna_tokenizer, dna_embeddings=None, img_embeddings=None, add_genus=True, max_length=1600):
        self.images = images
        self.dna_strings = np.array(dna_strings, dtype=np.str_)
        self.labels = np.array(labels, dtype=np.int64)
        self.img_processor = img_processor
        self.dna_tokenizer = dna_tokenizer
        self.dna_str_len_mapping = dna_str_len_mapping
        self.species2genus = species2genus - 1 if np.min(species2genus) == 1 else species2genus
        self.max_length = max_length
        self.genus_species = genus_species
        self.add_genus = add_genus
        
        self.dna_embeddings = dna_embeddings
        self.img_embeddings = img_embeddings

        self.v_dna_str_len = np.vectorize(_dna_str_len)
        self.v_dna_len_token = np.vectorize(lambda x: self.dna_str_len_mapping[x] if x in self.dna_str_len_mapping else -1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        dna_sequence = self.dna_strings[idx]
        if len(dna_sequence.shape) == 0:
            dna_sequence = np.array([dna_sequence])
        dna_sequence_len = np.array(list(map(self.v_dna_str_len, dna_sequence)))
        dna_len_token = np.array(list(map( self.v_dna_len_token, dna_sequence_len)))
        
        label = torch.tensor(self.labels[idx].reshape(-1, 1), dtype=torch.long) - 1

        res = {
            'dna_len_tokens': torch.tensor(dna_len_token, dtype=torch.long).reshape(-1, 1),
            'labels': label,
        }

        # ===== Image Processing =====
        # np.transpose(self.images[idx],(0, 2, 3, 1))

        if self.img_processor is not None:
            image = self.images[idx]
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            image = np.transpose(image,(0, 2, 3, 1))
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            

            image_inputs = self.img_processor(images=image, return_tensors="pt")

            res['image_inputs'] = image_inputs

        # ===== DNA Processing =====

        if self.dna_tokenizer is not None:
            dna_inputs = self.dna_tokenizer(
                dna_sequence.tolist(),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            res['dna_inputs'] = dna_inputs

        
        # ===== Label & Genus =====
        if self.add_genus:
            genus = torch.tensor(self.species2genus[label], dtype=torch.long).reshape(-1, 1)
            res['genus'] = genus

        if self.dna_embeddings is not None:
            res['dna_emb'] = torch.tensor(self.dna_embeddings[idx], dtype=torch.float32)
            if res['dna_emb'].ndim == 1:
                res['dna_emb'] = res['dna_emb'].unsqueeze(0)
        if self.img_embeddings is not None:
            res['img_emb'] = torch.tensor(self.img_embeddings[idx], dtype=torch.float32)
            if res['img_emb'].ndim == 1:
                res['img_emb'] = res['img_emb'].unsqueeze(0)

        return res
