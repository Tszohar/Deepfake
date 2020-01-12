import glob
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json

from torchvision.transforms import transforms


def create_samples_list(json_path: str, data_dir: str):
    """
    This function creates samples list with their label according to the files in data_dir.
    Every sample is identified as the full path of the file.
    :param json_path: file path to the labels json file
    :param data_dir: file path to the labels json file
    :return: This function returns samples list and their label
    """

    files_list = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    samples_list = []

    with open(json_path) as json_file:
        json_data = json.load(json_file)
    for filename in files_list:
        vid_folder = os.path.join(data_dir, filename)
        frames = glob.glob1(vid_folder, "*.jpg")
        file_label = json_data[filename + ".mp4"]['label']
        for frame in frames:
            img_full_path = os.path.join(vid_folder, frame)
            samples_list.append((img_full_path, file_label))
    return samples_list


class DFDataset(Dataset):
    def __init__(self, json_path: str, data_dir: str):
        """
        :param json_path: file path to the labels json file
        :param data_dir: files path
        """
        # TODO Guy CR: use private members and @property for getters
        self._data_dir = data_dir
        self._samples_list = create_samples_list(json_path=json_path, data_dir=data_dir)
        transforms_list = [transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ]
        self._transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self._samples_list)

    def __getitem__(self, item: int):
        label_index = 1
        image = self._transform(Image.open(self._samples_list[item][0]))
        if self._samples_list[item][label_index] == "FAKE":
            label = 1
        else:
            label = 0

        sample = {"image": image, "label": label, "label_str": self._samples_list[item][1]}

        return sample
