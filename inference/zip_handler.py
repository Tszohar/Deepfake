import os
import shutil
import zipfile

from inference import inf_config
from inference.video_handler import VideoPreProcess, VideoHandler


class ZipHandler:
    def __init__(self, input_zip):
        self._input_zip = input_zip
        self.extract()

    def extract(self):
        dst_folder = inf_config.videos_folder
        with zipfile.ZipFile(self._input_zip) as zip:
            if os.path.isdir(dst_folder):
                shutil.rmtree(dst_folder)

            os.makedirs(dst_folder)
            zip.extractall()

