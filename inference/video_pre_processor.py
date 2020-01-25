import os
import shutil
from typing import List, Callable

import cv2
import numpy as np
import torch
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

    def convert_to_frames(self,
                          video_file_path: str,
                          frame_decimation: int,
                          preprocessing_transform: Callable) -> List[torch.Tensor]:
        """
        :param video_file_path: video file path
        :param frame_decimation: video decimation to create frames
        :param preprocessing_transform: Transform that preprocesses the image
        :return: None
        """
        vidcap = cv2.VideoCapture(video_file_path)
        frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list = []
        for frame_idx in range(0, frame_num, frame_decimation):
            for i in range(frame_decimation):
                vidcap.grab()
            success, image = vidcap.retrieve()
            if not success:
                break
            assert image is not None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            img_cropped = self._frame_handler.process(image)
            if img_cropped is None:
                continue
            img_cropped = img_cropped / torch.abs(img_cropped).max()
            img_cropped = img_cropped / 2 + 0.5

            transformed_image = preprocessing_transform(img_cropped)
            frame_list.append(transformed_image)

        return frame_list

    @classmethod
    def dump_to_files(cls, frames: List[torch.Tensor], dst_folder: str) -> None:
        """
        Dumps list of frames to files
        :param frames: list of torch tensor, each corresponds to a frame in a video
        :param dst_folder: destination folder to save data in
        """
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)
        for idx, frame in enumerate(frames):
            np_image = frame.numpy()
            trans_image = np.transpose(np_image, [1, 2, 0])
            trans_image = (trans_image * 255)
            trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_folder, "frame{:03d}.jpg".format(idx)), trans_image)