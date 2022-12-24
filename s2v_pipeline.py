import os


target_app = 'com.desertstorm.recipebook'
target_app_json = '/trace_0/view_hierarchies/408.json'

#Get S2V embedding
print("Getting S2V embedding...")
os.system("python scripts/Screen2Vec/get_embedding.py -s scripts/Screen2Vec/filtered_traces/" + target_app + target_app_json+" -u 'scripts/Screen2Vec/UI2Vec_model.ep120' -m 'scripts/Screen2Vec/Screen2Vec_model_v4.ep120' -l 'scripts/Screen2Vec/layout_encoder.ep800'")
print("... Success.")

#Get Bert embedding
print("Getting Bert embedding...")
os.system("python scripts/bert/obert.py")
print("... Success.")


#Compute cosine similarity
print("Getting Cosine similarity...")
os.system("python scripts/exp1/similarity.py")
print("... Success.")

#Visualize cosine similarity
print("Visualizing Cosine similarity...")
os.system("python scripts/exp1/visualize.py")
print("... Success.")




