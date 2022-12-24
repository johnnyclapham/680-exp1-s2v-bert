import csv
import os
import sys

import torch
# import clip
from PIL import Image
from glob import glob
import evaluation_metrics as em
from transformers import BertTokenizer, BertModel

# from scripts.Screen2Vec.get_embedding import get_embedding



class OBQuery:
    def __init__(self, id, text, ground_truth):
        self.id = id
        self.text = text
        self.ground_truth = ground_truth


def getcossim(screen_path, ob_list, device,image_dict):
    print("screen_path "+str(screen_path))
    paths = glob(screen_path)
    paths.sort()
    print("ty")
    print(paths)
    
    each_app_result = []
    # print("ob list: "+ str(ob_list))
    for ob in ob_list:
        #change location 1, we can switch this text to be processed by bert.
        # text = clip.tokenize(ob.text, context_length=77, truncate=True).to(device)

        #vvvvv start bert code vvvvv
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        marked_text = "[CLS] " + ob.text + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,) # Whether the model returns all hidden-states.)
        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings_squeezed = torch.squeeze(token_embeddings, dim=1)
        #^^^^^^ end bert code ^^^^^^

        #compute similarity between this ob and the image embeddings in image_dict
        score = {}
        print("image_dict size: "+ str(len(image_dict)))
        for key, value in image_dict.items():
            print("key: "+str(key))
            img_id = os.path.splitext(key)[0]
            ob_gt = os.path.splitext(ob.ground_truth)[0]
            print("ob: "+str(id))
            print("img_id: "+str(img_id))
            print("ob_gt: "+str(ob_gt))
            print("image id is equal!!")
            image = value.to(device)
            image_flatten = torch.flatten(image.detach())
            text_flatten = torch.flatten(token_embeddings_squeezed)
            cos = torch.nn.CosineSimilarity(dim=0)
            s2vsize= image_flatten.shape[0]
            bertsize = text_flatten.shape[0]
            padding_length = (int(s2vsize)-int(bertsize))/2
            pad = torch.nn.ConstantPad1d(int(padding_length), 0)
            tensor_bert_flatten_padded = pad(text_flatten)
            output = cos(tensor_bert_flatten_padded, image_flatten)
            score[key] = output
            print("score: "+str(output))

        print("score dict: "+ str(score))
        ranked_screens = sorted(score.keys(), key=lambda f: -score[f])
        print(f'OB-ID: {ob.id}\tScore: {score}\t Ranked Screens: {ranked_screens}')

        each_ob_result = []
        for screen in ranked_screens:
            print("screen: "+str(screen))
            print("ob.ground_truth: "+str(ob.ground_truth))
            print("screen: "+str(screen))
            print("ob.ground_truth: "+str(ob.ground_truth))
            img_id = os.path.splitext(screen)[0]
            ob_gt = os.path.splitext(ob.ground_truth)[0]
            print("final img_id: "+str(img_id))
            print("final ob_gt: "+str(ob_gt))
            # if screen == ob.ground_truth:
            if img_id == ob_gt:
                each_ob_result.append(1)
            else:
                each_ob_result.append(0)

        each_app_result.append(each_ob_result)
        # print("ob.ground_truth: "+str(ob.ground_truth))
        print("each_ob_result: "+str(each_ob_result))
        print("ob.id: "+str(ob.id))
    return each_app_result


if __name__ == "__main__":
    ob_folder_path = '../data/obs_behaviors/test_set/'
    screen_folder_path = '../data/screens_per_app_test/'
    result_folder_path = '../results/result(Experiment-2-AllOB-SGD)-testset.csv'

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(result_folder_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['App-Name', '# of Queries', 'MRR', 'MAP', 'h@1', 'h@5', 'h@10'])

    app_name_list = []
    for app_name in os.listdir(ob_folder_path):
        print(app_name)
        if app_name == '.DS_Store':
            continue
        app_name_list.append(app_name)
    app_name_list.sort()
    print(f'Number of total app: {app_name_list.__len__()}')


    #Note: uncomment the full list for eval
    for i in range(0, len(app_name_list)):
        app_name = app_name_list[i]
        app_ob_path = ob_folder_path + app_name
        app_screens_path = screen_folder_path + app_name + '/*.json'
        result_of_each_app = []

        for root, dirs, files in os.walk(app_ob_path):
            ob_query_list = []
            for file in files:
                if not file.endswith(".csv"):
                    continue
                csv_file_path = root + '/' + file

                with open(csv_file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()[1:]
                    reader = csv.reader(lines, delimiter=';')
                    for row in reader:
                        if len(row) == 0:
                            continue
                        ob_id, ob_type, ob_text, screen_id, component_id = row[0], row[1], row[2], row[3], row[4]
                        ob_query = OBQuery(ob_id, ob_text, screen_id + '.jpg')
                        ob_query_list.append(ob_query)

        print(f'App name: {app_name}')
        print(f'# of OB: {ob_query_list.__len__()}')

        print(app_screens_path)
        image_dict = {}
        paths = glob(app_screens_path)
        print("paths: "+str(paths))
        for path in paths:
            image_id = path.split("/")[-1]
            #comput the image embeddings
            print("image id: "+ str(image_id))
            print('yam')
            #compute s2v embedding for image
            screen = path
            ui_model = 'scripts/Screen2Vec/UI2Vec_model.ep120'
            screen_model = 'scripts/Screen2Vec/Screen2Vec_model_v4.ep120'
            layout_model = 'scripts/Screen2Vec/layout_encoder.ep800'
            print("screen path: "+str(path))
            
            #Note: this is where the roadblock bug is
            # We cannot pass an imageto S2V, we need a json file.
            os.system("python scripts/Screen2Vec/get_embedding.py -s "+path+" -u "+ui_model+" -m "+screen_model+" -l "+layout_model+" -o '../s2vout'")
            
            #store embedding in image_dict
            image_dict[image_id] = torch.load('../s2vout/embeddings/S2V_screen_emb_test.pt')
            # image_dict[image_id] = tensorArray[0]
    
        print("image_dict: "+str(len(image_dict)))
        #if there are no screens for an app, skip it...
        if(len(image_dict)==0):
            continue
        result_of_each_app = getcossim(app_screens_path, ob_query_list, device,image_dict)
        print("result of each app"+str(result_of_each_app))
        # continue

        mrr = em.mean_reciprocal_rank(result_of_each_app)
        # print(f'MRR:{mrr}')
        map = em.mean_average_precision(result_of_each_app)
        # print(f'MAP:{map}')
        hit_1 = em.hit_rate_at_k(result_of_each_app, 1)
        # print(f'HIT@1:{hit_1}')
        hit_5 = em.hit_rate_at_k(result_of_each_app, 5)
        # print(f'HIT@1:{hit_5}')
        hit_10 = em.hit_rate_at_k(result_of_each_app, 10)
        # print(f'HIT@1:{hit_10}')

        with open(result_folder_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([app_name, result_of_each_app.__len__(), mrr, map, hit_1, hit_5, hit_10])

        # print(f'# of apps completed: {i}\n')
        print('# of apps completed: '+str(i+1))
