import scipy.io
from matplotlib import pyplot as plt
import numpy as np


MAT_FILE_PATH = '/mnt/sdb4/insect_dataset.mat'

mat = scipy.io.loadmat(MAT_FILE_PATH)

mat['all_images']
# type(mat['all_images'])
# mat['all_images'][0]

# plt.imshow(mat[0])
# i = mat[0]
# i
# i = mat['all_images'][0]
# plt.imshow(i)
# i
# i[0]
# i[1]
# i[2]
# mat
# mat.keys()
# mat
# mat.keys()
# mat['all_image_features_gan']
# mat['all_image_features_gan'][0]
# mat.keys()
# mat['all_dna_features_cnn_original']
# mat.keys()
# mat['all_dnas']
# dnas = mat['all_dnas']
# dnas.argmax(2)
# dnas = dnas.argmax(2)
# dnas
# dnas[0].tolist()[:5]
# dnas[0].tolist()[:10]
# dnas[0].tolist()[:20]
# np.unique(dnas[0])

# np.unique(dnas[0])
# np.unique(dnas)
# mat['all_labels']
# mat['all_labels'][0]
# mat['all_labels'][0][10]
# mat['all_labels'][0][22]
# np.unique(mat['all_labels'][0])
# mat['all_labels'][0]
# mat
# mat.keys()
# mat['all_boldids'][0]
# mat['all_boldids']
# mat.keys()
# mat['described_species_labels_train']
# mat.keys()
# mat['all_string_dnas']
# mat['all_string_dnas'][0]
# len(mat['all_string_dnas'][0])
# map(len, mat['all_string_dnas'])
# list(map(len, mat['all_string_dnas']))
# list(map(lambda s: len(s.strip()), mat['all_string_dnas']))
# set(list(map(lambda s: len(s.strip()), mat['all_string_dnas'])))
# # Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# # Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
