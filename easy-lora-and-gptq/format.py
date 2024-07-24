import glob
import os
import shutil

src_path = "./tiny-imagenet-200/"
train_dst_path = "./train/"
test_dst_path = "./test/"
split_ratio = 0.9

def check_folder_and_make(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    check_folder_and_make(train_dst_path)
    check_folder_and_make(test_dst_path)
    folders = os.listdir(src_path)
    for folder in folders:
        check_folder_and_make(train_dst_path+folder)
        check_folder_and_make(test_dst_path+folder)
        img_paths = glob.glob(src_path+folder+"/images/*.JPEG")

        img_len = len(img_paths)
        train_index = int(img_len * split_ratio)

        train_set = img_paths[:train_index]
        test_set = img_paths[train_index:]

        for train in train_set:
            shutil.copy2(train, train_dst_path+folder)

        for test in test_set:
            shutil.copy2(test, test_dst_path+folder)

if __name__ == "__main__":
    main()