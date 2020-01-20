import PIL
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN


class FrameHandler:
    """
    FrameHandler is responsible to send each frame to MTCNN with a given properties,
    which crops the face from the image
    """
    def __init__(self, image_size: int):
        self._cropper = MTCNN(image_size=image_size, margin=int(0.3 * image_size), device=torch.device("cuda"))

    # TODO: Add output type hinting
    def process(self, frame: PIL.Image):
        return self._cropper(frame)
