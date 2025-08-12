# %%
import torch
from torch import nn
import torchviz
# import dnabert
# import vit
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import dataset
import dnabert
from importlib import reload
from sklearn.decomposition import PCA

# %%
MAT_FILE_PATH = './insect_dataset.mat'

mat = scipy.io.loadmat(MAT_FILE_PATH)
mat.keys()


# %%
pca = PCA(n_components=512)
all_dna_features_cnn_pca = pca.fit_transform(mat['all_dna_features_cnn_new'])
# %%
pca = PCA(n_components=512)
all_image_features_gan_pca = pca.fit_transform(mat['all_image_features_gan'])

# %%
import models
reload(models)
from models import AttentionFusion, SpiciePredictor, Decoder
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
fusion = AttentionFusion(512, 512, 512).to(device)
decoder = Decoder(512, 1050).to(device)
predictor = SpiciePredictor(fusion, decoder).to(device)
predictor.fit(all_dna_features_cnn_pca, 
              all_image_features_gan_pca, 
              mat['all_labels'].squeeze(), 
              mat['val_seen_loc'].squeeze(), 
              mat['train_loc'].squeeze(), 
              5000, 
              lr=0.0005,
              device=device)
