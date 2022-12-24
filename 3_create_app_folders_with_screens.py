import os
import shutil
import csv


def read_csv(path):
    dict_from_csv = {}
    with open(path, 'r') as f:
        next(f)
        reader = csv.reader(f, delimiter=',')
        for ui_number, app_name, a, b in reader:
            if app_name in dict_from_csv:
                # dict_from_csv[app_name].append(ui_number + '.jpg')
                dict_from_csv[app_name].append(ui_number + '.json')
            else:
                # dict_from_csv[app_name] = [ui_number + '.jpg']
                dict_from_csv[app_name] = [ui_number + '.json']
    # print(dict_from_csv)
    return dict_from_csv


def create_folder_with_screens(dict_from_csv, read_folder_path, write_folder_path):
    counter = 0
    for application_name, ui_number_list in dict_from_csv.items():
        destination_path = write_folder_path + '/' + application_name
        print("destination_path: "+str(destination_path))
        os.mkdir(destination_path)
        for root, directory, files in os.walk(read_folder_path):
            for file in files:
                print("destination_path: "+str(destination_path))
                print("root: "+str(root+"  file: "+file))
                print("ui_number_list: "+str(ui_number_list))
                if file in ui_number_list:
                    shutil.copy(os.path.join(root, file), destination_path)
                    # source = os.path.join(root, file)
                    print("source: "+os.path.join(root, file))
                    # os.system("scp "+source+" "+destination_path)

        counter += 1
        # print(f'Number of completed app: {counter}')


# This program creates different folders for each app where each folder contains the screens (jpg) of that app
if __name__ == '__main__':
    rico_ui_details_path = './ui_details.csv'
    filtered_screens_path = '../data/filtered_json_files'
    output_folder_path = '../data/screens_per_app_test'

    dictionary_from_csv = read_csv(rico_ui_details_path)
    print(dictionary_from_csv)

    create_folder_with_screens(dictionary_from_csv, filtered_screens_path, output_folder_path)
