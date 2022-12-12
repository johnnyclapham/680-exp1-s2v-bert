from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
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
# tensor_bert = torch.from_numpy(tensor_bert)
# tensor_s2v = torch.from_numpy(tensor_s2v)
# tensor_s2v = tf.nn.l2_normalize(tensor_s2v, 0)
# tensor_bert = tf.nn.l2_normalize(tensor_bert, 0)
print(type(tensor_bert))
print(type(tensor_s2v))
tensor_bert_array = tensor_bert.numpy()
tensor_s2v_array = tensor_s2v.numpy()
print(type(tensor_bert_array))
print(type(tensor_s2v_array))
# print(tensor_bert_array.shape)
# print(tensor_bert_array)
tensor1 = tf.convert_to_tensor(tensor_bert_array)
tensor2 = tf.convert_to_tensor(tensor_s2v_array)
print(type(tensor_bert_array))
print(type(tensor_s2v_array))
output = cos(tensor1, tensor2)




print('end')
