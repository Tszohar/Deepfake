import torch

input_zip = "/media/guy/Files 3/kaggle_competitions/deepfake/test_videos.zip"
model = "/media/guy/Files 3/kaggle_competitions/deepfake/results/model_epoch_9.pth"
results_dir = "/media/guy/Files 3/kaggle_competitions/deepfake/results"
videos_folder = "/media/guy/Files 3/kaggle_competitions/deepfake/test_videos"
frames_folder = "/media/guy/Files 3/kaggle_competitions/deepfake/test_frames"

batch_size = 128
image_size = 224
frames_decimation = 10

device = torch.device("cuda")