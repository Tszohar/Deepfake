from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter


class IMetric(ABC):
    @abstractmethod
    def calculate(self, gt_labels: np.ndarray, predicted_probs: np.ndarray, predicted_labels: np.ndarray) -> float:
        raise NotImplementedError


class LogLoss(IMetric):
    def calculate(self, gt_labels: np.ndarray, predicted_probs: np.ndarray, predicted_labels: np.ndarray) -> float:
        return metrics.log_loss(y_true=gt_labels, y_pred=predicted_probs, eps=1e-5, labels=[1, 0])


class Accuracy(IMetric):
    def calculate(self, gt_labels: np.ndarray, predicted_probs: np.ndarray, predicted_labels: np.ndarray) -> float:
        return metrics.accuracy_score(y_true=gt_labels, y_pred=predicted_labels)


class Precision(IMetric):
    def calculate(self, gt_labels: np.ndarray, predicted_probs: np.ndarray, predicted_labels: np.ndarray) -> float:
        return metrics.precision_score(y_true=gt_labels, y_pred=predicted_labels, labels=[1, 0], pos_label=0)


class Recall(IMetric):
    def calculate(self, gt_labels: np.ndarray, predicted_probs: np.ndarray, predicted_labels: np.ndarray) -> float:
        return metrics.recall_score(y_true=gt_labels, y_pred=predicted_labels, labels=[1, 0], pos_label=0)


class MetricsCalculator:
    def __init__(self, writer: SummaryWriter, metrics: List[IMetric]):
        self._writer = writer
        self._iterations = 0

        self._gt_labels = []
        self._predicted_probs = []
        self._loss = []

        self._metrics = metrics

    def update_metrics(self, loss: np.ndarray, gt_labels: np.ndarray, predicted_probs: np.ndarray):
        self._gt_labels.append(gt_labels)
        self._predicted_probs.append(predicted_probs)
        self._loss.append(loss)

        self._writer.add_scalar(tag='loss', scalar_value=loss, global_step=self._iterations)
        self._iterations += 1

    def finish_epoch(self, epoch: int):
        gt_labels = np.concatenate(self._gt_labels)
        predicted_probs = np.concatenate(self._predicted_probs)
        predicted_labels = np.argmax(self._predicted_probs, axis=1)

        for metric in self._metrics:
            metric_val = metric.calculate(gt_labels=gt_labels,
                                          predicted_probs=predicted_probs,
                                          predicted_labels=predicted_labels)
            metric_name = metric.__class__.__name__
            print("epoch {} validation {} is: {}".format(epoch, metric_name, metric_val))
            self._writer.add_scalar(tag=metric_name, scalar_value=metric_val, global_step=epoch)

        self._predicted_probs.clear()
        self._gt_labels.clear()
        self._loss.clear()
