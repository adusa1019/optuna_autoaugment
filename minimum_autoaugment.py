from random import randint

import chainer
import optuna
from chainer import functions as F
from chainer import links as L
from chainer import training
from chainer.training import extensions

BATCHSIZE = 128
EPOCH = 10


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Preprocess(chainer.dataset.DatasetMixin):

    def __init__(self, data, v_flip=False, h_flip=False, debug=False):
        self.data = data
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.debug = debug

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        image, label = self.data[i]
        image = image.reshape([28, 28])

        if self.v_flip and randint(0, 1):
            image = image[::-1]

        if self.h_flip and randint(0, 1):
            image = image[:, ::-1]

        if self.debug:
            from PIL import Image
            im = Image.fromarray(image * 255)
            im.show()

        return image.flatten(), label


def objective(trial):
    model = L.Classifier(MLP(1000, 10))

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    v_flip = trial.suggest_categorical("vertical flip", [True, False])
    h_flip = trial.suggest_categorical("horizontal flip", [True, False])
    mnist = chainer.datasets.get_mnist()
    train = Preprocess(mnist[0], v_flip, h_flip)
    test = Preprocess(mnist[1])
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    test_iter = chainer.iterators.SerialIterator(test, BATCHSIZE, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (EPOCH, 'epoch'), './results')
    trainer.extend(extensions.Evaluator(test_iter, model))
    log_report_extension = extensions.LogReport(log_name=None)
    trainer.extend(
        extensions.PrintReport([
            "epoch", "main/loss", "validation/main/loss", "main/accuracy",
            "validation/main/accuracy"
        ]),
        trigger=(1, 'epoch'))
    trainer.extend(log_report_extension)

    trainer.run()

    log_last = log_report_extension.log[-1]
    for key, value in log_last.items():
        trial.set_user_attr(key, value)
    val_err = 1.0 - log_report_extension.log[-1]["validation/main/accuracy"]
    return val_err


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print(study.best_trial)
