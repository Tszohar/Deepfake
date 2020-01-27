import glob
import os
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json

from torchvision.transforms import transforms


def create_samples_list(json_path: str, data_dir: str) -> List[Tuple[str, str]]:
    """
    This function creates samples list with their label according to the files in data_dir.
    Every sample is identified as the full path of the file.
    :param json_path: file path to the labels json file
    :param data_dir: file path to the labels json file
    :return: This function returns samples list and their label
    """

    files_list = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    fake_samples = []
    real_samples = []

    with open(json_path) as json_file:
        json_data = json.load(json_file)
    for filename in files_list:
        vid_folder = os.path.join(data_dir, filename)
        frames = glob.glob1(vid_folder, "*.jpg")
        file_label = json_data[filename + ".mp4"]['label']
        for frame in frames:
            img_full_path = os.path.join(vid_folder, frame)
            if file_label == "FAKE":
                fake_samples.append((img_full_path, file_label))
            else:
                real_samples.append((img_full_path, file_label))
    return real_samples + fake_samples


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

    def class_sample_count(self) -> List[int]:
        fake_samples = sum(sample[1] == "FAKE" for sample in self._samples_list)
        real_samples = sum(sample[1] == "REAL" for sample in self._samples_list)
        return [real_samples, fake_samples]
