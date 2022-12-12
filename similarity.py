from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
# cos_sim = cosine_similarity(vector1,vector2)

tensor_bert = tf.io.read_file(
    './bert_embeddings_unsqueezed.pt', name=None
)

tensor_s2v = tf.io.read_file(
    './S2V_embeddings.pt', name=None
)

tensor_bert = tf.convert_to_tensor(tensor_bert)
tensor_s2v = tf.convert_to_tensor(tensor_s2v)
cos = torch.nn.CosineSimilarity(dim=0)
print(type(tensor_bert))
print(type(tensor_s2v))
output = cos(tensor_bert, tensor_s2v)

print('yam')
