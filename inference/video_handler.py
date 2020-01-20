import glob
import os
import shutil

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch import nn
from torchvision import models, transforms

from inference import inf_config


class VideoPreProcessor:
    """
    VideoPreProcessor converts each video file to frames according to a certain configuration
    """
    def __init__(self, image_size: int):
        self._frame_handler = FrameHandler(image_size=image_size)
        self.dst_folder = None

    def convert_to_frames(self, video_file_path: str, frame_decimation: int):
        """
        :param video_file_path: video file path
        :param root_folder: root directory to save the output frames for video_file_path
        :param frame_decimation: video decimation to create frames
        :return: None
        """
        dst_folder = os.path.splitext(video_file_path)[0]
        self.dst_folder = dst_folder
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


class FrameHandler:
    """
    FrameHandker is responsible to send each frame to MTCNN with a given properties,
    which crops the face from the image
    """
    def __init__(self, image_size:int):
        super().__init__()
        device = torch.device("cuda")
        self._cropper = MTCNN(image_size=image_size, margin=int(0.3 * image_size), device=device)
        self._faces_list = None

    def process(self, frame: PIL.Image):
        return self._cropper(frame)


def load_config():
    """
    loading needed configuration for the processed video
    """
    frame_decimation = inf_config.frames_decimation
    image_size = inf_config.image_size
    return image_size, frame_decimation


class VideoHandler:
    """
    VideoHandler receives video file path in order to detect the probability of being 'FAKE'
    """
    def __init__(self):

        self._image_size, self._frame_decimation = load_config()
        self._cropped_frames_folder = None
        self.pre_process_video = VideoPreProcessor(image_size=self._image_size)
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              ])
        self._net = self.load_net()
        self._results = []
        self._softmax = nn.Softmax()
        self.evaluation = None

    def handle(self, video_file_path: str):
        """
        :param video_file_path: video file path to process
        :return: This function returns the probability of video_file_path being 'FAKE'
        """
        self.pre_process_video.convert_to_frames(video_file_path=video_file_path,
                                                 frame_decimation=self._frame_decimation)
        self._cropped_frames_folder = self.pre_process_video.dst_folder
        fake_probability = self.detect()
        return fake_probability


    def load_net(self):
        """
        :return: This function loads pre trained neural network: ResNet34
        """
        resnet34 = models.resnet34(pretrained=False)
        resnet34.fc = nn.Linear(512, 2)
        model = torch.load(inf_config.model)
        resnet34.load_state_dict(model, strict=False)
        device = torch.device("cuda")
        resnet34.to(device)
        resnet34.eval()
        return resnet34

    def detect(self):
        """
        :return: This function calculates Fake probability for each frame in self._cropped_frames_folder
         and returns the mean probability
        """
        frames_list = glob.glob1(self._cropped_frames_folder, "*.jpg")
        device = torch.device("cuda")

        for frame in frames_list:
            frame_path = os.path.join(self._cropped_frames_folder, frame)
            image = self._transform(PIL.Image.open(frame_path))
            output = self._net(image[None, ::].to(device))
            softmax = self._softmax(output)
            fake_probability = softmax[0][1].item()
            self._results.append(fake_probability)
        return self.majority_vote()

    def majority_vote(self):
        """
        :return: return the mean probability of self._results list
        """
        return np.mean(self._results)
