import torch

train_data_path = "/media/guy/Files 3/kaggle_competitions/deepfake/processed_data"
train_json_path = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_0/metadata.json"
validation_data_path = "/media/guy/Files 3/kaggle_competitions/deepfake/processed_data_validation"
validation_json_path = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_1/metadata.json"
pretrained_weights_path = "/media/guy/Files 3/kaggle_competitions/deepfake/resnet34/resnet34-333f7ec4.pth"
results_dir = "/media/guy/Files 3/kaggle_competitions/deepfake/results"

experiment_name = "balanced_data"

batch_size = 128
epoch_size = 100
image_size = 224

device = torch.device("cuda")