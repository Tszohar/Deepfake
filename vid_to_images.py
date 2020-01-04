import os
import shutil
import time
from logging import Logger

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torch import nn

from log import get_logger


def convert_vid_to_images(logger: Logger, video_path: str, detector: nn.Module = None):
    logger.info("Starting converting video: {}".format(video_path))
    file_name = os.path.basename(video_path)
    dirname = os.path.dirname(video_path)
    dst_folder = os.path.join(dirname, os.path.splitext(file_name)[0])

    image_size = 150
    if detector is None:
        detector = MTCNN(image_size=image_size, margin=int(0.3*image_size), device="cuda:0")

    if os.path.isdir(dst_folder):
        shutil.rmtree(dst_folder)

    os.makedirs(dst_folder)

    start_time = time.time()
    vidcap = cv2.VideoCapture(file_path)

    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        img_cropped = detector(image)
        np_image = img_cropped.numpy()
        trans_image = np.transpose(np_image, [1, 2, 0])
        trans_image = trans_image / np.max(np.abs(trans_image))
        trans_image = trans_image / 2 + 0.5
        trans_image = (trans_image * 255)
        trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dst_folder, "frame{:03d}.jpg".format(count)), trans_image)
        count += 1

    elapsed_time = time.time() - start_time
    logger.info("Conversion done (elapsed: {:.2f})".format(elapsed_time))


if __name__ == "__main__":
    mini_data_path = "/media/guy/Files 3/kaggle_competitions/deepfake/analyze"
    file_name = "apzckowxpy.mp4"
    file_path = os.path.join(mini_data_path, file_name)
    logger = get_logger()
    convert_vid_to_images(video_path=file_path, detector=None, logger=logger)