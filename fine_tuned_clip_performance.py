import csv
import os
import sys

import torch
import clip
from PIL import Image
from glob import glob
import evaluation_metrics as em


class OBQuery:
    def __init__(self, id, text, ground_truth):
        self.id = id
        self.text = text
        self.ground_truth = ground_truth


def get_screen_ranking(screen_path, ob_list, model, preprocess, device):
    model = model.to(device)
    paths = glob(screen_path)
    paths.sort()
    image_dict = {}
    for path in paths:
        image_id = path.split("/")[-1]
        image_dict[image_id] = preprocess(Image.open(path)).unsqueeze(0)

    each_app_result = []
    for ob in ob_list:
        text = clip.tokenize(ob.text, context_length=77, truncate=True).to(device)
        score = {}
        for key, value in image_dict.items():
            image = value.to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                score[key] = logits_per_image.item()

        ranked_screens = sorted(score.keys(), key=lambda f: -score[f])
        # print(f'OB-ID: {ob.id}\tScore: {score}\t Ranked Screens: {ranked_screens}')

        each_ob_result = []
        for screen in ranked_screens:
            if screen == ob.ground_truth:
                each_ob_result.append(1)
            else:
                each_ob_result.append(0)

        each_app_result.append(each_ob_result)
    return each_app_result


if __name__ == "__main__":
    # These paths are used for testing
    # ob_folder_path = '/Users/antusaha/Documents/GitHub/bug_report_mapping/RICO_Data/ir_engine_data/input_data/splitted_data/testing_data/csv_files/'
    # ob_folder_path = './clip_performance_data_test/ob/'
    # screen_folder_path = './clip_performance_data_test/screens/'
    # result_folder_path = './clip_performance_data_test/output/v2_last_update_ViT-L_14.csv'

    # These paths are used for running on bg1 machine
    # ob_folder_path = '/home/asaha02/Documents/Dataset/RICO_Data/splitted_data/testing_data/csv_files/'
    # screen_folder_path = '/home/asaha02/Documents/Dataset/RICO_Data/Screens_Per_App/'
    # result_folder_path = '/home/asaha02/Documents/Result/Raw_CLIP_Result/raw_clip_result_bg1(1088_to_rest).csv'

    # These paths are used for running on our server
    ob_folder_path = '../data/obs_behaviors/test_set/'
    screen_folder_path = '../data/full_images/'
    result_folder_path = '../results/result(Experiment-2-AllOB-SGD).csv'

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, preprocess = clip.load("ViT-B/32")

    model.load_state_dict(torch.load("Experiment-2-AllOB-SGD.pt")['model_state_dict'])
    model.eval()

    with open(result_folder_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['App-Name', '# of Queries', 'MRR', 'MAP', 'h@1', 'h@5', 'h@10'])

    app_name_list = []
    for app_name in os.listdir(ob_folder_path):
        if app_name == '.DS_Store':
            continue
        app_name_list.append(app_name)
    app_name_list.sort()
    print(f'Number of total app: {app_name_list.__len__()}')
    # sys.exit()

    for i in range(0, len(app_name_list)):
        app_name = app_name_list[i]

        app_ob_path = ob_folder_path + app_name
        app_screens_path = screen_folder_path + app_name + '/*.jpg'
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
        result_of_each_app = get_screen_ranking(app_screens_path, ob_query_list, model, preprocess, device)

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

        print(f'# of apps completed: {i}\n')
