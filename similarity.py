from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
# cos_sim = cosine_similarity(vector1,vector2)
torch.set_printoptions(threshold=10_000)

#Note: Load our tensors from the saved files
tensor_bert = torch.load('./bert_embeddings_squeezed.pt')
tensor_s2v = torch.load('./S2V_screen_emb.pt')
#Note: initialize our cosine similarity function
cos = torch.nn.CosineSimilarity(dim=0)
#Note: compute the cosine similarity between the two tensors
output = cos(tensor_bert, tensor_s2v)
torch.save(output, 'cosine_similarity.pt')
with open('cosine_similarity.txt', 'w') as f:
    f.write(str(output))

print(output)






#debuging print statements below
# print(type(tensor_bert))
# print(type(tensor_s2v))
# print('____________________________')
# print(type(tensor_bert))
# print(type(tensor_s2v))
# # tensor_s2v = torch.stack(tensor_s2v, dim=0)
# print(type(tensor_bert))
# print(type(tensor_s2v))
# output = cos(tensor_bert, tensor_s2v)
# print(output)
print('end')
