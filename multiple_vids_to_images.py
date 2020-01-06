import glob
import multiprocessing as mp
import os
import shutil
import time
from _queue import Empty
from logging import Logger

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from log import get_logger


class VideoConverter(mp.Process):
    def __init__(self, input_q: mp.Queue, stop_signal: mp.Value, out_folder: str, frame_decimation: int):
        super().__init__()
        self._detector = None
        self._input_q = input_q
        self._stop_signal = stop_signal
        self._out_folder = out_folder
        self.busy_signal = mp.Value('b')
        self.daemon = True
        self._frame_decimation = frame_decimation

    @property
    def detector(self):
        if self._detector is None:
            image_size = 150
            self._detector = MTCNN(image_size=image_size, margin=int(0.3 * image_size), device="cuda:0")
        return self._detector

    def run(self) -> None:
        logger = get_logger()
        self.busy_signal.value = True
        while not self._stop_signal.value:
            try:
                file_path = self._input_q.get(block=True, timeout=0.1)
                self.busy_signal.value = True
                self.convert_video_to_images(logger=logger,
                                             video_path=file_path)
            except Empty:
                self.busy_signal.value = False

    def convert_video_to_images(self, logger: Logger, video_path: str):
        if not os.path.isfile(video_path):
            logger.info("File not found: {}".format(video_path))
            return

        logger.info("Starting converting video: {}".format(video_path))
        file_name = os.path.basename(video_path)
        dst_folder = os.path.join(self._out_folder, os.path.splitext(file_name)[0])

        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)

        os.makedirs(dst_folder)

        start_time = time.time()
        vidcap = cv2.VideoCapture(video_path)

        count = 0
        while True:
            image = None
            for i in range(self._frame_decimation):
                success, image = vidcap.read()
                if not success:
                    elapsed_time = time.time() - start_time
                    logger.info("Conversion done (elapsed: {:.2f})".format(elapsed_time))
                    return
            assert image is not None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            img_cropped = self.detector(image)
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


if __name__ == "__main__":
    input_folder = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_0"
    output_folder = "/media/guy/Files 3/kaggle_competitions/deepfake/processed_data"

    file_names = glob.glob1(input_folder, "*.mp4")

    process_num = 8
    input_q = mp.Queue(maxsize=20)
    stop_signal = mp.Value('b')
    stop_signal.value = False
    video_converters = [VideoConverter(input_q=input_q,
                                       stop_signal=stop_signal,
                                       out_folder=output_folder,
                                       frame_decimation=10)
                        for idx in range(process_num)]
    for vid_conv in video_converters:
        vid_conv.start()

    for file_name in file_names:
        input_q.put(os.path.join(input_folder, file_name))

    while any(vid_conv.busy_signal.value for vid_conv in video_converters) or input_q.qsize() > 0:
        time.sleep(0.1)
