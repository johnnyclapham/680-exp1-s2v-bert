# from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import Tensor
import numpy as np
import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
# cos_sim = cosine_similarity(vector1,vector2)
torch.set_printoptions(threshold=10_000)

#Note: Load our tensors from the saved files
tensor_bert = torch.load('./embeddings/bert_embeddings_squeezed.pt')
tensor_s2v = torch.load('./embeddings/S2V_screen_emb.pt')
print(tf.shape(tensor_bert))

# tensor_bert_expanded = tensor_bert.expand(tensor_s2v.size())
# Note: flatten both tensors to be one dimension
#        This can avoid the 2d similarity
tensor_bert_flatten = torch.flatten(tensor_bert)
tensor_s2v_flatten = torch.flatten(tensor_s2v.detach())

print(tf.shape(tensor_bert))
print(tf.shape(tensor_bert_flatten))
# print(tf.shape(tensor_s2v))
print(tf.shape(tensor_s2v_flatten))


#Note: initialize our cosine similarity function
cos = torch.nn.CosineSimilarity(dim=0)
# print(tf.losses.CosineSimilarity()(tensor_bert,tensor_s2v))
#Note: compute the cosine similarity between the two tensors
output = cos(tensor_bert, tensor_s2v)
torch.save(output, 'embeddings/cosine_similarity.pt')
with open('embeddings/cosine_similarity.txt', 'w') as f:
    f.write(str(output))

# print("Cosine Similarity:",output)
# print(output)
# s2vsize= tensor_s2v_flatten.shape[0]
# bertsize = tensor_bert_flatten.shape[0]
# padding_length = (int(s2vsize)-int(bertsize))/2
# print(type(padding_length))
# pad = torch.nn.ConstantPad1d(padding_length, 0)
# pad = torch.nn.ConstantPad1d(int(padding_length), 0)


#Note: Initialize cosine sim function
cossin = torch.nn.CosineSimilarity(dim=0)

#Note: cos function requires tensors of same size
#       we will pad the smaller (bert) tensor to be 
#       the same size as the larger (s2v) tensor
s2vsize= tensor_s2v_flatten.shape[0]
bertsize = tensor_bert_flatten.shape[0]
padding_length = (int(s2vsize)-int(bertsize))/2
pad = torch.nn.ConstantPad1d(int(padding_length), 0)
tensor_bert_flatten_padded = pad(tensor_bert_flatten)

# print("tensor_bert_flatten_padded: "+str(tensor_bert_flatten_padded.shape[0]))
# print("tensor_s2v_flatten: "+str(tensor_s2v_flatten.shape[0]))

output = cossin(tensor_s2v_flatten, tensor_bert_flatten_padded)

# output.reshape(1, 1)
print(output)

# fig = output.visualize_heatmap()
# fig.write_html("heatmap.html")






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
