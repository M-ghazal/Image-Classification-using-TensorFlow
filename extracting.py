import os
import shutil
import random

negative_data_path = r"Negative data path"
positive_data_path = r"Positive data path"

train_dir = r"Train data path"
test_dir = r"Test data path"

train_negative_dir = os.path.join(train_dir, "Negative")
train_positive_dir = os.path.join(train_dir, "Positive")
test_negative_dir = os.path.join(test_dir, "Negative")
test_positive_dir = os.path.join(test_dir, "Positive")

os.makedirs(train_negative_dir, exist_ok=True)
os.makedirs(train_positive_dir, exist_ok=True)
os.makedirs(test_negative_dir, exist_ok=True)
os.makedirs(test_positive_dir, exist_ok=True)

def split_data(src_path, train_dest, test_dest, split_ratio=0.7):
    files = os.listdir(src_path)
    random.shuffle(files)

    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    test_files = files[split_point:]

    for file_name in train_files:
        shutil.copy(os.path.join(src_path, file_name), os.path.join(train_dest, file_name))

    for file_name in test_files:
        shutil.copy(os.path.join(src_path, file_name), os.path.join(test_dest, file_name))

split_data(negative_data_path, train_negative_dir, test_negative_dir)
split_data(positive_data_path, train_positive_dir, test_positive_dir)

print("Data split into training and testing sets successfully.")
