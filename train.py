import datetime
import os
from collections import OrderedDict

import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import DFDataset
from metrics import MetricsCalculator, LogLoss, Accuracy, Precision, Recall


def my_load_state_dict(net: nn.Module, saved_model: OrderedDict):
    net_state_dict = net.state_dict()
    new_state_dict = OrderedDict([(name, value)
                                  for name, value in saved_model.items()
                                  if name in net_state_dict and net_state_dict[name].shape == value.shape])

    net.load_state_dict(new_state_dict, strict=False)


def get_dataset(json_path: str, data_path: str) -> DFDataset:
    dataset = DFDataset(json_path=json_path, data_dir=data_path)
    return dataset


def get_model(pretrained_path: str):
    resnet34 = models.resnet34(pretrained=False)
    saved_model = torch.load(pretrained_path)
    # Changing last fc layer to become binary output
    resnet34.fc = nn.Linear(512, 2)
    my_load_state_dict(net=resnet34, saved_model=saved_model)
    resnet34.to(config.device)
    return resnet34


def get_optimizer(model: nn.Module):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    criterion.to(config.device)
    return criterion


def run_validation(net: models, dataloader: DataLoader, epoch: int, metrics_calc: MetricsCalculator):
    net.eval()
    validation_run_counter = 0
    criterion = get_criterion()
    softmax = nn.Softmax(dim=1)

    for i_batch, sample_batched in enumerate(dataloader):
        validation_run_counter += 1
        outputs = net(sample_batched['image'].to(config.device))
        loss_validation = criterion(outputs, sample_batched['label'].to(config.device))
        metrics_calc.update_metrics(loss=loss_validation.numpy(),
                                    gt_labels=sample_batched['label'].detach().cpu().numpy(),
                                    predicted_probs=softmax(outputs).detach().cpu().numpy())
    metrics_calc.finish_epoch(epoch=epoch)


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
    net = get_model(config.pretrained_weights_path)
    optimizer = get_optimizer(model=net)
    criterion = get_criterion()
    softmax = nn.Softmax(dim=1)

    run_counter = 0

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_validation = SummaryWriter(log_dir=os.path.join(log_dir, 'validation'))

    metrics = [LogLoss(), Accuracy(), Precision(), Recall()]
    metrics_train = MetricsCalculator(writer=writer_train, metrics=metrics)
    metrics_validation = MetricsCalculator(writer=writer_validation, metrics=metrics)

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

            metrics_train.update_metrics(loss=loss.detach().cpu().numpy(),
                                         gt_labels=sample_batched['label'].detach.cpu().numpy(),
                                         predicted_probs=softmax(outputs).detach().cpu().numpy())

            if i_batch % 5 == 0:
                print('[epoch: {}, batch: {}] loss: {:.3f}'.format(epoch + 1, i_batch + 1, loss.item()))
                running_loss = 0.0
            writer_train.file_writer.flush()

        metrics_train.finish_epoch(epoch=epoch)

        run_validation(net=net, dataloader=validation_dataloader, epoch=epoch, metrics_calc=metrics_validation)
        save_model(net=net, epoch=epoch, output_dir=log_dir)


if __name__ == "__main__":
    train_dataset = get_dataset(json_path=config.train_json_path, data_path=config.train_data_path)
    class_sample_count = np.array(train_dataset.class_sample_count(), dtype=np.int)
    target = torch.cat((torch.zeros(class_sample_count[0], dtype=torch.long),
                        torch.ones(class_sample_count[1], dtype=torch.long)))
    weights = 1. / class_sample_count

    samples_weight = torch.tensor([weights[t] for t in target])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=samples_weight, num_samples=len(train_dataset))
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, sampler=sampler)
    validation_dataset = get_dataset(json_path=config.validation_json_path, data_path=config.validation_data_path)
    validation_dl = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    run_train(train_dataloader=train_dl, validation_dataloader=validation_dl)
