import glob
import os
import shutil

import PIL
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch import nn
from torchvision import models, transforms

from inference import inf_config


class VideoPreProcess:
    def __init__(self, video_path: str):
        file_name = os.path.splitext(os.path.basename(video_path))
        self._frame_decimation = inf_config.frames_decimation
        self.convert_to_frames(video=video_path, file_name=file_name[0])

    def convert_to_frames(self, video: str, file_name: str):
        root_folder = inf_config.frames_folder
        dst_folder = os.path.join(root_folder, os.path.splitext(file_name)[0])
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)

        vidcap = cv2.VideoCapture(video)

        count = 0
        while True:
            image = None
            for i in range(self._frame_decimation):
                success, image = vidcap.read()
                if not success:
                    return
            assert image is not None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            img_cropped = FrameHandler(image)
            if img_cropped.faces_list is None:
                continue
            np_image = img_cropped.faces_list.numpy()
            trans_image = np.transpose(np_image, [1, 2, 0])
            trans_image = trans_image / np.max(np.abs(trans_image))
            trans_image = trans_image / 2 + 0.5
            trans_image = (trans_image * 255)
            trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_folder, "frame{:03d}.jpg".format(count)), trans_image)
            count += 1


class FrameHandler(PIL.Image.Image):
    def __init__(self, frame: PIL.Image.Image):
        super().__init__()
        image_size = inf_config.image_size
        device = inf_config.device
        self._frame = frame
        self._cropper = MTCNN(image_size=image_size, margin=int(0.3 * image_size), device=device)
        self.faces_list = self.identify_faces()

    def identify_faces(self):
        self.faces_list = self._cropper(self._frame)
        return self.faces_list


class VideoHandler:
    def __init__(self, video_file_path: str):
        root_folder = inf_config.frames_folder
        single_video_folder = os.path.splitext(os.path.split(video_file_path)[1])[0]
        VideoPreProcess(video_path=video_file_path)
        self._cropped_frames_folder = os.path.join(root_folder, single_video_folder)
        self.image_to_tensor()
        self.net = self.load_net()
        self._results = pd.DataFrame()
        self.detect()

    def load_net(self):
        resnet34 = models.resnet34(pretrained=False)
        resnet34.fc = nn.Linear(512, 2)
        model = torch.load(inf_config.model)
        resnet34.load_state_dict(model, strict=False)
        resnet34.to(inf_config.device)
        resnet34.eval()
        return resnet34

    def detect(self):
        frames_list = glob.glob1(self._cropped_frames_folder, "*.jpg")
        device = inf_config.device
        for frame in frames_list:
            frame_path = os.path.join(self._cropped_frames_folder, frame)
            image = self._transform(PIL.Image.open(frame_path))
            output = self.net(image[None, ::].to(device))
            self._results = self._results.append({'label': output[0,output.argmax()]}, ignore_index=True)
        self.majority_vote()

    def majority_vote(self):
        if np.sum(self._results['label']) / len(self._results) > 0:
            self.evaluation = 'FAKE'
        else:
            self.evaluation = 'REAL'

    def image_to_tensor(self):
        transforms_list = [transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ]
        self._transform = transforms.Compose(transforms_list)