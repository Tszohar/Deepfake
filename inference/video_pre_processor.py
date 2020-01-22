import os
import shutil

import cv2
import numpy as np
from PIL import Image

from inference.frame_handler import FrameHandler


class VideoPreProcessor:
    """
    VideoPreProcessor converts each video file to frames according to a certain configuration
    """
    def __init__(self, image_size: int, output_path: str):
        self._frame_handler = FrameHandler(image_size=image_size)
        self._working_folder = output_path

    @classmethod
    def get_dst_folder(cls, video_file_path: str) -> str:
        return os.path.splitext(video_file_path)[0]

    def convert_to_frames(self, video_file_path: str, frame_decimation: int) -> None:
        """
        :param video_file_path: video file path
        :param root_folder: root directory to save the output frames for video_file_path
        :param frame_decimation: video decimation to create frames
        :return: None
        """
        dst_folder = self._working_folder
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)

        vidcap = cv2.VideoCapture(video_file_path)

        count = 0
        while True:
            image = None
            for i in range(frame_decimation):
                success, image = vidcap.read()
                if not success:
                    return
            assert image is not None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            img_cropped = self._frame_handler.process(image)
            if img_cropped is None:
                continue
            np_image = img_cropped.numpy()
            trans_image = np.transpose(np_image, [1, 2, 0])
            trans_image = trans_image / np.max(np.abs(trans_image))
            trans_image = trans_image / 2 + 0.5
            trans_image = (trans_image * 255)
            trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_folder, "frame{:03d}.jpg".format(count)), trans_image)
            count += 1