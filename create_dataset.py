import glob
import os
from torch.utils.data import Dataset
import json


def create_list(json_path, data_dir):
    videos = os.listdir(data_dir)
    samples_list = []
    with open(json_path) as json_file:
        data = json.load(json_file)
        for vid in videos:
            vid_folder = os.path.join(data_dir, vid)
            frames = glob.glob1(vid_folder, "*.jpg")
            for frame in frames:
                img_full_path = os.path.join(vid_folder, frame)
                samples_list.append((img_full_path, data[vid + ".mp4"]['label']))
    return samples_list


class DFDataset(Dataset):
    def __init__(self, json_path, data_dir):
        self.data_dir = data_dir
        self.data = create_list(json_path=json_path, data_dir=data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == "__main__":
    path = "/media/guy/Files 3/kaggle_competitions/deepfake/processed_data"
    json_path = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_0/metadata.json"
    ds = DFDataset(json_path, path)
    print("bla")