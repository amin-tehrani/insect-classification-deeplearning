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
mat.keys()

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
# genus = lambda s: mat['species2genus'][s]-1
# genus(156)

# %%
# group labels count
# pd.Series(mat['all_labels'].squeeze()).value_counts()


# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=512)
all_dna_features_cnn_pca = np.array(pca.fit_transform(mat['all_dna_features_cnn_new']))


# %%
pca = PCA(n_components=768)
all_image_features_gan_pca = np.array(pca.fit_transform(mat['all_image_features_gan']))

# %%
# mat['species2genus'].shape

# %%
# x = np.array(list(map(lambda s: len(s.strip()), mat['all_string_dnas'])))
# np.unique(x).size

# %%

all_dna_len = list(map(lambda s: len(s.strip()), mat['all_string_dnas']))
dna_str_len_mapping: dict[int,int] = {}

def dna_str_len_to_int(s_len):
    if s_len not in dna_str_len_mapping:
        dna_str_len_mapping[s_len] = len(dna_str_len_mapping)
    return dna_str_len_mapping[s_len]

# def all_dna_len_token():
#     return list(map(dna_str_len_to_int, all_dna_len))

all_dna_len_tokens = list(map(dna_str_len_to_int, all_dna_len))
all_dna_len_tokens = np.array(all_dna_len_tokens, dtype=np.int64)
# print(list(zip(all_dna_len, all_dna_len_tokens)))

# %%
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deviceCPU = torch.device("cpu")

device = deviceCPU
print("Device:", device)

from multimodal_dataset import MultiModalDataset
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
dna_tokenizer, dna_encoder = get_tokenizer_encoder("./dnaencoder-finetuned1755100772-final", device)
# get_dna_embedding(mat['all_string_dnas'][:2], dna_tokenizer, dna_encoder).shape

# %%
dataset = MultiModalDataset(mat['all_string_dnas'], mat['all_images'], np.transpose(mat['all_labels'], (1,0)), dna_str_len_mapping, species2genus, genus_species, img_processor, dna_tokenizer, 
                            dna_embeddings=all_dna_features_cnn_pca, img_embeddings=all_image_features_gan_pca)

# %%
from models import AttentionFusion, GenusClassifier, LocalSpecieClassfier, MainClassifier
from transformers import DefaultDataCollator

fusion_embedder = AttentionFusion(dna_dim=512,img_dim=768,dna_len_dim=16)
print("Fusion model created. fused dim: ", fusion_embedder.fused_dim)
genus_classifier = GenusClassifier(fusion_embedder.fused_dim)

local_specie_classifier = LocalSpecieClassfier(fusion_embedder.fused_dim)

main_classifier = MainClassifier(mat['species2genus'], genus_species, dna_encoder, img_encoder, fusion_embedder, genus_classifier, local_specie_classifier).to(device)

# for (x,d) in zip(all_dna_len_tokens, dataset):
#     print(x,"\t" ,d['dna_len_tokens'], d['labels'],d['genus'])
# from models import multimodal_collator
# print(main_classifier(**multimodal_collator([dataset[0], dataset[1]])))
# print(main_classifier(**dataset[0]))



train_indices = (mat['train_loc'] - 1).flatten()  # Get train indices
val_indices = (mat['val_seen_loc'] - 1).flatten()  # Get validation indices

train_dataset = MultiModalDataset(mat['all_string_dnas'][train_indices], mat['all_images'][train_indices], np.transpose(mat['all_labels'], (1,0))[train_indices], dna_str_len_mapping, species2genus, genus_species, img_processor, dna_tokenizer, dna_embeddings=all_dna_features_cnn_pca[train_indices], img_embeddings=all_image_features_gan_pca[train_indices])
val_dataset = MultiModalDataset(mat['all_string_dnas'][val_indices], mat['all_images'][val_indices], np.transpose(mat['all_labels'], (1,0))[val_indices], dna_str_len_mapping, species2genus, genus_species, img_processor, dna_tokenizer, dna_embeddings=all_dna_features_cnn_pca[val_indices], img_embeddings=all_image_features_gan_pca[val_indices])

# # import os
# # os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
# # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# # # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(main_classifier.fit(train_dataset, val_dataset, batch_size=8))


# x = genus_classifier(dataset[0:2]['dna_len_token'].unsqueeze(0).to(deviceGPU), 
#                 get_dna_embedding(dataset[0:2]['dna_len_token'].unsqueeze(0).to(deviceGPU), 
#                                   get_img_embedding(dataset[0:2]['pixel_values'].unsqueeze(0).to(deviceGPU), img_processor, img_encoder)))
# print(x.shape, x)


# specie_predictor = MainClassifier(mat['species2genus'],genus_species, genus_classifier).to(deviceCPU)
# specie_predictor.fit(
#     all_dna_len_tokens,
#     all_dna_features_cnn_pca, 
#     all_image_features_gan_pca, 
#     mat['all_labels'].squeeze(), 
#     mat['val_seen_loc'].squeeze(), 
#     mat['train_loc'].squeeze(), 
#     200, 
#     lr=0.005,
#     eval_frequency=10,
#     freeze_genus=True,
#     teacher_force=True,
#     device=deviceCPU)
