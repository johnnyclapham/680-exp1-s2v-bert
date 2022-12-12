from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
# cos_sim = cosine_similarity(vector1,vector2)

# tensor_bert = tf.io.read_file('./bert_embeddings_unsqueezed.pt', name=None)
tensor_bert = torch.load('./bert_embeddings_unsqueezed.pt')

# tensor_s2v = tf.io.read_file('./S2V_embeddings.pt', name=None)
tensor_s2v = torch.load('./S2V_embeddings.pt')


cos = torch.nn.CosineSimilarity(dim=0)


print(type(tensor_bert))
print(type(tensor_s2v))


print('____________________________')
print(type(tensor_bert))
print(type(tensor_s2v))
output = cos(tensor_bert, tensor_bert)
print(output)




print('end')
