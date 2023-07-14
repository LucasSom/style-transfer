import os

import pandas as pd
from tensorflow import keras


class PrintLearningRate(keras.callbacks.Callback):
    """Print the learning rate if it changes."""

    def __init__(self):
        super().__init__()
        self.lr = 0

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = self.model.optimizer.learning_rate.numpy()
        if lr != self.lr:
            self.lr = lr
            print("----------------------------- CAMBIÓ LEARNING RATE -----------------------------")
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, self.lr))


class IncrementKLBeta(keras.callbacks.Callback):
    """Increment value of kl_beta by a ratio

    Arguments:
        initial: initial value of kl_beta
        ratio: ratio to increment
    """

    def __init__(self, initial, ratio, threshold):
        super().__init__()
        self.kl_beta = initial
        self.ratio = ratio
        self.threshold = threshold

    def on_epoch_begin(self, epoch, logs=None):
        if self.kl_beta < self.threshold:
            self.model.get_layer('kl_beta').variables[0].assign(self.kl_beta)

    def on_epoch_end(self, epoch, logs=None):
        self.kl_beta += self.ratio


class LossHistory(keras.callbacks.Callback):
    """
    Save values of loss and accuracy on a CSV

    Arguments:
        log_path: path where save the CSV file
    """

    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {'loss': [None], 'accuracy': [None]}

        if os.path.isfile(self.log_path) and epoch > 0:
            prev_callbacks = pd.read_csv(self.log_path)
        else:
            prev_callbacks = pd.DataFrame()
            with open(self.log_path, 'w'):
                pass
        d = {k: [v] for k, v in logs.items()}
        d['epoch'] = 0
        new_callbacks = prev_callbacks.append(pd.DataFrame(d))

        new_callbacks.to_csv(self.log_path)
        print(f"Guardado el csv en {self.log_path}")
