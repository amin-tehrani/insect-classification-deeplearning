# %%
import torch
from torch import nn
import torchviz

import vit
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from importlib import reload
from mat import mat


# %%
species2genus = mat['species2genus']-1

# group species by genus

genus_species = dict()
max_specie_in_genus = 0
for genus_id, genus in pd.DataFrame(species2genus, columns=['genus']).groupby('genus'):
    specie_indices = genus.index.tolist()
    genus_species[genus_id] = specie_indices
    if len(specie_indices) > max_specie_in_genus:
        max_specie_in_genus = len(specie_indices)

print(len(genus_species))
print("Max specie in genus: ", max_specie_in_genus)


# %%

all_dna_len = list(map(lambda s: len(s.strip()), mat['all_string_dnas']))
dna_str_len_mapping: dict[int,int] = {}

def dna_str_len_to_int(s_len):
    if s_len not in dna_str_len_mapping:
        dna_str_len_mapping[s_len] = len(dna_str_len_mapping)
    return dna_str_len_mapping[s_len]


all_dna_len_tokens = list(map(dna_str_len_to_int, all_dna_len))
all_dna_len_tokens = np.array(all_dna_len_tokens, dtype=np.int64)

# %%
deviceGPU = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
deviceCPU = torch.device("cpu")

device = deviceGPU
print("Device:",device)

# %%
import vit
reload(vit)
from vit import get_processor_encoder, get_img_embedding
img_processor, img_encoder = get_processor_encoder("./vit-finetuned7-final", device)
# get_img_embedding(mat['all_images'][:2], img_processor, img_encoder, device).shape

# %%
import dnaencoder
reload(dnaencoder)
from dnaencoder import get_tokenizer_encoder, get_dna_embedding
dna_tokenizer, dna_encoder = get_tokenizer_encoder("./dnaencoder-finetuned1755100772-final", deviceGPU)
# get_dna_embedding(mat['all_string_dnas'][:2], dna_tokenizer, dna_encoder).shape


from load_embeddings import load_dna_embeddings, load_img_embeddings

all_dna_features = load_dna_embeddings()
all_image_features = load_img_embeddings()


# %%
train_indices = (mat['train_loc'] - 1).flatten()  # Get train indices
val_indices = np.concatenate([
    (mat['val_seen_loc'] - 1).flatten(),
    (mat['val_unseen_loc'] - 1).flatten()
])  # Get validation indices (seen + unseen)

# %%
import multimodal_dataset
reload(multimodal_dataset)
from multimodal_dataset import MultiModalDataset

train_dataset = MultiModalDataset(mat['all_string_dnas'][train_indices], mat['all_images'][train_indices], np.transpose(mat['all_labels'], (1,0))[train_indices], dna_str_len_mapping, species2genus, genus_species, None, None, dna_embeddings=all_dna_features[train_indices], img_embeddings=all_image_features[train_indices])
val_dataset = MultiModalDataset(mat['all_string_dnas'][val_indices], mat['all_images'][val_indices], np.transpose(mat['all_labels'], (1,0))[val_indices], dna_str_len_mapping, species2genus, genus_species, None, None, dna_embeddings=all_dna_features[val_indices], img_embeddings=all_image_features[val_indices])
# %%
import models
reload(models)
from models import AttentionFusion, AttentionFusionV2, GenusClassifier, LocalSpecieClassfier, MainClassifier, multimodal_collector

def get_main_classifier():
    fusion_embedder = AttentionFusion(dna_dim=512,img_dim=768,dna_len_dim=32, fused_dim=256, proj_dna_dim=128-32, proj_img_dim=128, dropout=0.2)
    # fusion_embedder = AttentionFusionV2(dna_dim=512,img_dim=768,dna_len_dim=32, fused_dim=256, dropout=0.2)
    print("Fusion model created. fused dim: ", fusion_embedder.fused_dim)
    genus_classifier = GenusClassifier(fusion_embedder.fused_dim,dropout=0.2, dna_len_dim=32)

    local_specie_classifier = LocalSpecieClassfier(fusion_embedder.fused_dim,reduced_fused_dim=128, specie_decoder_hidden_dim=256, dropout=0.2,dna_len_dim=32)

    return MainClassifier(mat['species2genus'], genus_species, None, None, fusion_embedder, genus_classifier,
                          local_specie_classifier,
                          alpha=2, beta=0, theta=0,
                          ).to(device)

main_classifier = get_main_classifier()

import warnings
warnings.filterwarnings("ignore")

# %%
main_classifier.fit(
    train_dataset,
    val_dataset,
    batch_size=512,
    epochs=500,
    eval_steps=200,
    save_steps=400,
    lr=0.0001
)


