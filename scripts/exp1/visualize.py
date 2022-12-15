from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

cs_tensor = torch.load('./embeddings/cosine_similarity.pt')
cs_tensor = cs_tensor.detach().numpy()
# plt.figure(1, figsize=(2^13, 2^13))
plt.figure(1, figsize=(2^13,2^3))
plt.imshow(cs_tensor, cmap='hot',aspect='auto',vmin=0, interpolation='nearest')
# plt.imshow(cs_tensor, cmap='hot',aspect='equal',vmin=0, interpolation='nearest')

# plt.show()
plt.title("Cosine Similarity Heatmap")
plt.ylabel("OB Embedding")
plt.xlabel("Screen Embedding")
plt.savefig('cs_hd.png', dpi=300)
plt.show()


# cs_tensor = torch.load('./cosine_similarity.pt')
# cs_tensor = cs_tensor.detach().numpy()
# plt.figure(1, figsize=(4, 60))
# plt.imshow(cs_tensor, cmap='hot', interpolation='nearest')
# plt.show()
# # plt.savefig('cs.png')