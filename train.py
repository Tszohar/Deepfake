import datetime
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DFDataset
import torchvision.models as models
import config
from sklearn.metrics import log_loss, accuracy_score

def get_dataset(json_path: str, data_path: str):
    dataset = DFDataset(json_path=json_path, data_dir=data_path)
    return dataset


def get_model():
    pretrained_weights_path = config.pretrained_weights_path
    resnet34 = models.resnet34(pretrained=False)
    model = torch.load(pretrained_weights_path)
    resnet34.load_state_dict(model, strict=False)
    optimizer = torch.optim.Adam(resnet34.parameters(), lr=1e-3)
    # Changing last fc layer to become binary output
    resnet34.fc = nn.Linear(512, 2)
    resnet34.to(config.device)
    return resnet34, optimizer


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    criterion.to(config.device)
    return criterion


def run_validation(net: models, dataloader: DataLoader, log_dir: str, epoch: int):
    net.eval()
    validation_run_counter = 0
    writer_validation = SummaryWriter(log_dir=os.path.join(log_dir, 'validation'))
    criterion = get_criterion()
    softmax = nn.Softmax(dim=1)

    batch_log_loss = np.zeros((len(dataloader), 1))
    batch_accuracy = np.zeros((len(dataloader), 1))

    for i_batch, sample_batched in enumerate(dataloader):
        validation_run_counter += 1
        outputs = net(sample_batched['image'].to(config.device))
        loss_validation = criterion(outputs, sample_batched['label'].to(config.device))
        writer_validation.add_scalar(tag='loss', scalar_value=loss_validation.item(),
                                     global_step=validation_run_counter)
        writer_validation.file_writer.flush()
        batch_log_loss[i_batch] = log_loss(y_true=sample_batched['label'].detach().cpu(),
                                           y_pred=softmax(outputs)[:, 1].detach().cpu(), eps=1e-5, labels=[1,0])
        batch_accuracy[i_batch] = accuracy_score(y_true=sample_batched['label'].detach().cpu(),
                                                 y_pred=softmax(outputs).argmax(axis=1).detach().cpu())
    val_log_loss = np.mean(batch_log_loss)
    val_accuracy = np.mean(batch_accuracy)
    print("epoch {} validation log_loss is: {}".format(epoch, val_log_loss))
    print("epoch {} validation accuracy is: {}".format(epoch, val_accuracy))
    writer_validation.add_scalar(tag='log_loss', scalar_value=val_log_loss, global_step=epoch)
    writer_validation.add_scalar(tag='accuracy', scalar_value=val_accuracy, global_step=epoch)


def save_model(net: nn.Module, epoch: int, output_dir: str):
    torch.save(net.state_dict(), os.path.join(output_dir, 'model_epoch_{}.pth'.format(epoch)))


def run_train(train_dataloader: DataLoader, validation_dataloader: DataLoader):
    training_start_time = datetime.datetime.now()
    results_dir = config.results_dir
    experiment_name = config.experiment_name
    log_dir = os.path.join(results_dir, "{}_{}".format(training_start_time.strftime("%Y%m%d (%H:%M:%S.%f)"),
                                                       experiment_name))
    epoch_size = config.epoch_size
    device = config.device
    net, optimizer = get_model()
    criterion = get_criterion()
    softmax = nn.Softmax(dim=1)

    run_counter = 0
    batch_log_loss = np.zeros((len(train_dataloader), 1))
    batch_accuracy = np.zeros((len(train_dataloader), 1))

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    for epoch in range(epoch_size):
        running_loss = 0.0
        net.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            run_counter += 1
            optimizer.zero_grad()
            outputs = net(sample_batched['image'].to(device))
            loss = criterion(outputs, sample_batched['label'].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer_train.add_scalar(tag='loss', scalar_value=loss.item(), global_step=run_counter)
            if i_batch % 5 == 0:
                print('[epoch: {}, batch: {}] loss: {:.3f}'.format(epoch + 1, i_batch + 1, loss.item()))
                running_loss = 0.0
            writer_train.file_writer.flush()

            batch_log_loss[i_batch] = log_loss(y_true=sample_batched['label'].detach().cpu(),
                                               y_pred=softmax(outputs)[:, 1].detach().cpu(), eps=1e-5, labels=[1, 0])
            batch_accuracy[i_batch] = accuracy_score(y_true=sample_batched['label'].detach().cpu(),
                                                     y_pred=softmax(outputs).argmax(axis=1).detach().cpu())
        epoch_log_loss = np.mean(batch_log_loss)
        epoch_accuracy = np.mean(batch_accuracy)
        print("epoch {} log_loss is: {}".format(epoch, epoch_log_loss))
        print("epoch {} accuracy is: {}".format(epoch, epoch_accuracy))
        batch_log_loss = np.zeros((len(train_dataloader), 1))
        batch_accuracy = np.zeros((len(train_dataloader), 1))

        writer_train.add_scalar(tag='log_loss', scalar_value=epoch_log_loss, global_step=epoch)
        writer_train.add_scalar(tag='accuracy', scalar_value=epoch_accuracy, global_step=epoch)

        run_validation(net=net, dataloader=validation_dataloader, log_dir=log_dir, epoch=epoch)
        save_model(net=net, epoch=epoch, output_dir=log_dir)


if __name__ == "__main__":
    train_dataset = get_dataset(json_path=config.train_json_path, data_path=config.train_data_path)
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    validation_dataset = get_dataset(json_path=config.validation_json_path, data_path=config.validation_data_path)
    validation_dl = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    run_train(train_dataloader=train_dl, validation_dataloader=validation_dl)
