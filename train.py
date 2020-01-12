import datetime
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from create_dataset import DFDataset
import torchvision.models as models

if __name__ == "__main__":

    train_data_path = "/media/guy/Files 3/kaggle_competitions/deepfake/processed_data"
    train_json_path = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_0/metadata.json"
    validation_data_path = "/media/guy/Files 3/kaggle_competitions/deepfake/processed_data_validation"
    validation_json_path = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_1/metadata.json"
    pretrained_weights_path = "/media/guy/Files 3/kaggle_competitions/deepfake/resnet34/resnet34-333f7ec4.pth"
    log_dir = "/media/guy/Files 3/kaggle_competitions/deepfake/results"

    training_date = datetime.datetime.now()
    experiment_name = "test1"
    log_dir = os.path.join(log_dir, "{}_{}".format(training_date.strftime("%Y%m%d (%H:%M:%S.%f)"), experiment_name))

    batch_size = 32
    epoch_size = 10
    device="cuda"

    train_dataset = DFDataset(json_path=train_json_path, data_dir=train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_dataset = DFDataset(json_path=validation_json_path, data_dir=validation_data_path)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    resnet34 = models.resnet34(pretrained=False)
    resnet34.to(device)
    model = torch.load(pretrained_weights_path)
    resnet34.load_state_dict(model, strict=False)
    optimizer = torch.optim.Adam(resnet34.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Pre trained models expects normalized images with a shape of as least 224
    ## Ask Guy
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Changing last fc layer to become binary output
    resnet34.fc = nn.Linear(512, 2)
    run_counter = 0

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_validation = SummaryWriter(log_dir=os.path.join(log_dir, 'validation'))

    for epoch in range(epoch_size):
        running_loss = 0.0
        resnet34.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            run_counter += 1
            optimizer.zero_grad()
            outputs = resnet34(sample_batched['image'].to(device))
            loss = criterion(outputs, sample_batched['label'].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer_train.add_scalar(tag='loss', scalar_value=loss.item(), global_step=run_counter)
            if i_batch % 5 == 0:
                print('[epoch: {}, batch: {}] loss: {:.3f}'.format(epoch + 1, i_batch + 1, loss.item()))
                running_loss = 0.0
            writer_train.file_writer.flush()


        ## Validation ##
        resnet34.eval()
        loss_validation = 0.0
        validation_run_counter = 0
        for i_batch, sample_batched in enumerate(validation_dataloader):
            outputs = resnet34(sample_batched['image'].to(device))
            loss_validation = criterion(outputs, sample_batched['label'].to(device))
            writer_validation.add_scalar(tag='loss', scalar_value=loss.item(), global_step=validation_run_counter)
            writer_validation.file_writer.flush()

        torch.save(resnet34.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))





    print("bla")