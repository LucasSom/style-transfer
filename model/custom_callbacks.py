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

    def __init__(self, initial, ratio):
        super().__init__()
        self.kl_beta = initial
        self.ratio = ratio

    def on_epoch_begin(self, epoch, logs=None):
        self.model.get_layer('kl_beta').variables[0].assign(self.kl_beta)

    def on_epoch_end(self, epoch, logs=None):
        self.kl_beta += self.ratio
