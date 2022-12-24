import os
import shutil

# This program copies the RICO json files (each json file corresponds to a app screen)
# that are listed in filter_file_path
if __name__ == '__main__':
    # source_path = '/home/asaha02/Documents/Dataset/RICO_Data/combined'
    # source_path = '../data/combined copy'
    source_path = '../data/combined'
    # destination_path = '/home/asaha02/Documents/Dataset/RICO_Data/Filtered_jpg_files'
    destination_path = '../data/filtered_json_files'

    # source_path = '../rico_data/combined'
    # destination_path = '../rico_data/Filtered_JSON_Files'
    filter_file_path = './filter_24598.txt'

    os.makedirs(destination_path, exist_ok=True)

    with open(filter_file_path, 'r') as f:
        filter_file = f.readlines()

    print("filter_file length:" +str(len(filter_file)))
    # exit()
    filter_file_list = []
    for i in filter_file:
        filter_file_list.append(i.replace('\n', ''))
    print("filter_file_list length:" +str(len(filter_file_list)))
    # exit()
    image_file_list = []
    for i in filter_file_list:
        # image_file_list.append(i.replace('.json', '.jpg'))
        image_file_list.append(i)

    # print(image_file_list)

    for root, directory, files in os.walk(source_path):
        for file in files:
            if file in image_file_list:
                print(os.path.join(root, file))
                shutil.copy(os.path.join(root, file), destination_path)
