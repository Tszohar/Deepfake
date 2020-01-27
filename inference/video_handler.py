from typing import List

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from inference.video_pre_processor import VideoPreProcessor


class VideoHandler:
    """
    VideoHandler receives video file path in order to detect the probability of being 'FAKE'
    """
    def __init__(self, image_size: int, frame_decimation: int, output_path: str, model_path: str):
        self._image_size = image_size
        self._frame_decimation = frame_decimation
        self._working_folder = output_path

        self.video_pre_processor = VideoPreProcessor(image_size=self._image_size, output_path=self._working_folder)

        self._transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              ])
        self._net = self.load_net(model_path=model_path, device=torch.device("cuda"))

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

    def handle(self, video_file_path: str) -> float:
        """
        :param video_file_path: video file path to process
        :return: This function returns the probability of video_file_path being 'FAKE'
        """
        frame_list = self.video_pre_processor.convert_to_frames(
            video_file_path=video_file_path,
            frame_decimation=self._frame_decimation,
            preprocessing_transform=self._transform
        )
        fake_probability = self.classify(frame_list=frame_list)
        return fake_probability

    def classify(self, frame_list: List[torch.Tensor]) -> float:
        """
        :return: This function calculates Fake probability for each frame in self._cropped_frames_folder
         and returns the mean probability
        """
        if len(frame_list) == 0:
            return 0.5

        device = torch.device("cuda")
        frames = torch.stack(frame_list, dim=0)
        output = self._net(frames.to(device))
        probabilities = nn.Softmax(dim=0)(torch.mean(output, dim=0)).detach().cpu().numpy()

        video_fake_prob = float(probabilities[1])

        eps = 1e-8
        video_fake_prob = max(eps, min(1-eps, video_fake_prob))

        return video_fake_prob
