import os


target_app = 'com.desertstorm.recipebook'
target_app_json = '/trace_0/view_hierarchies/408.json'

#Get S2V embedding
os.system("python scripts/Screen2Vec/get_embedding.py -s scripts/Screen2Vec/filtered_traces/" + target_app + target_app_json+" -u 'scripts/Screen2Vec/UI2Vec_model.ep120' -m 'scripts/Screen2Vec/Screen2Vec_model_v4.ep120' -l 'scripts/Screen2Vec/layout_encoder.ep800'")

#Get Bert embedding
os.system("python scripts/bert/obert.py")

# #Compute cosine similarity
os.system("python scripts/exp1/similarity.py")

# #Visualize cosine similarity
os.system("python scripts/exp1/visualize.py")



