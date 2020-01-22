import glob
import os

import PIL
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from inference import inf_config
from inference.video_pre_processor import VideoPreProcessor


class VideoHandler:
    """
    VideoHandler receives video file path in order to detect the probability of being 'FAKE'
    """
    def __init__(self, image_size: int, frame_decimation: int, output_path: str):
        self._image_size = image_size
        self._frame_decimation = frame_decimation
        self._working_folder = output_path

        self.video_pre_processor = VideoPreProcessor(image_size=self._image_size, output_path=self._working_folder)

        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              ])
        self._net = self.load_net(model_path=inf_config.model, device=torch.device("cuda"))
        self._softmax = nn.Softmax()

    def handle(self, video_file_path: str) -> float:
        """
        :param video_file_path: video file path to process
        :return: This function returns the probability of video_file_path being 'FAKE'
        """
        self.video_pre_processor.convert_to_frames(video_file_path=video_file_path,
                                                   frame_decimation=self._frame_decimation)
        # cropped_frames_folder = self.video_pre_processor.get_dst_folder(video_file_path)
        fake_probability = self.classify(frames_folder=self._working_folder)
        return fake_probability

    def load_net(self, model_path: str, device: torch.device) -> nn.Module:
        """
        :return: This function loads pre trained neural network: ResNet34
        """
        resnet34 = models.resnet34(pretrained=False)
        resnet34.fc = nn.Linear(512, 2)
        model = torch.load(model_path)
        resnet34.load_state_dict(model, strict=False)
        resnet34.to(device)
        resnet34.eval()
        return resnet34

    def classify(self, frames_folder: str) -> float:
        """
        :return: This function calculates Fake probability for each frame in self._cropped_frames_folder
         and returns the mean probability
        """
        frames_list = glob.glob1(frames_folder, "*.jpg")
        device = torch.device("cuda")

        frames_result = []
        for frame in frames_list:
            frame_path = os.path.join(frames_folder, frame)
            image = self._transform(PIL.Image.open(frame_path))
            output = self._net(image[None, ::].to(device))
            softmax = self._softmax(output)
            fake_probability = softmax[0][1].item()
            frames_result.append(fake_probability)

        video_prob = self.calc_video_prob(np.array(frames_result))

        return video_prob

    def calc_video_prob(self, probs: np.ndarray) -> float:
        """
        :return: return the mean probability of self._results list
        """
        return float(np.mean(probs))
