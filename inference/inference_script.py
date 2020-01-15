import glob
import os
import multiprocessing as mp
import zipfile

import torch
import torchvision.models as models
import time
from torch.utils.data import DataLoader
import pandas as pd
from dataset import DFDataset
from inference import inf_config
from inference.video_handler import VideoHandler
from inference.zip_handler import ZipHandler
from multiple_vids_to_images import VideoConverter






if __name__ == "__main__":
    zip_file_path = inf_config.input_zip
    print("bla")
   # ZipHandler(zip_file_path)
    videos_list = glob.glob1(inf_config.videos_folder, "*.mp4")
    for vid in videos_list:
        vid_file_path = os.path.join(inf_config.videos_folder, vid)
        result = VideoHandler(video_file_path=vid_file_path)
        print('{} is {}'.format(vid, result.evaluation))


