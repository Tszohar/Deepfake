import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

image_path = "/media/guy/Files 3/kaggle_competitions/deepfake/analyze/apzckowxpy/frame015.jpg"
img = Image.open(image_path)

image_size = 150
mtcnn = MTCNN(image_size=image_size, margin=int(0.3*image_size))
img_cropped = mtcnn(img)

np_image = img_cropped.numpy()
trans_image = np.transpose(np_image, [1, 2, 0])
trans_image = trans_image / np.max(np.abs(trans_image))
trans_image = trans_image / 2 + 0.5

plt.imshow(trans_image)
plt.show()
