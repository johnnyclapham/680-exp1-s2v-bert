import torch
from transformers import BertTokenizer, BertModel
import numpy
import logging
import matplotlib.pyplot as plt
import tensorflow as tf

# tf.compat.v1.disable_v2_behavior()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "The shopping list button is broken."
# with open('ob.txt', 'r') as file:
with open('text/ob.txt', 'r') as file:
    ob = file.read().replace('\n', '')
marked_text = "[CLS] " + ob + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
#print (tokenized_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

#print (segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# print(tokens_tensor)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,) # Whether the model returns all hidden-states.)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers. 
with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]

# print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
# layer_i = 0

# print ("Number of batches:", len(hidden_states[layer_i]))
# batch_i = 0

# print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
# token_i = 0

# print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full")
token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings.size()
token_embeddings_squeezed = torch.squeeze(token_embeddings, dim=1)
# print('____________________________')
# print(numpy.shape(token_embeddings))
# print(numpy.shape(token_embeddings_squeezed))

# Note: Save as tensors
# torch.save(token_embeddings_squeezed, 'bert_embeddings_squeezed.pt')
# # torch.save(token_embeddings, 'bert_embeddings_unsqueezed.pt')
print('saving...')
torch.save(token_embeddings_squeezed, 'embeddings/bert_embeddings_squeezed.pt')
# torch.save(embeddings/token_embeddings, 'bert_embed

# print(token_embeddings.dtype)
# print(type(token_embeddings))

# with open('bert_embeddings_squeezed.txt', 'w') as f:
#     f.write(str(token_embeddings_squeezed))

# with open('bert_embeddings_unsqueezed.txt', 'w') as f:
#     f.write(str(token_embeddings))



