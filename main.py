import gzip
import sys
import pandas as pd
import numpy as np
import torch
from Model import Model
import time
from Visualization import PlotConfusionMatrix, convert, ShowExamples

image_size = 28
from sklearn.metrics import confusion_matrix

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def read_data():
    files_images = []
    files_images.append('train-images-idx3-ubyte.gz')
    files_images.append('t10k-images-idx3-ubyte.gz')
    files_labels = []
    files_labels.append('train-labels-idx1-ubyte.gz')
    files_labels.append('t10k-labels-idx1-ubyte.gz')
    images = []
    list_labels = []
    for file_images, file_labels in zip(files_images, files_labels):
        f = gzip.open(file_images, 'r')
        f.read(16)
        buf = f.read()
        f.close()
        num_images = sys.getsizeof(buf) // (image_size * image_size)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X = data.reshape(num_images, image_size * image_size) / 255
        images.append(X)

        f = gzip.open(file_labels, 'r')
        f.read(8)
        buf = f.read()
        f.close()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels = labels.reshape(num_images)
        list_labels.append(labels)
    return images[0], list_labels[0], images[1], list_labels[1]


def accuary(model, x, y, name):
    loss = []
    size_batch = 1024
    dummies_y = pd.get_dummies(y)
    for fnum in range(0, len(dummies_y), size_batch):
        temp_loss = model.loss(x[fnum:fnum + size_batch], dummies_y[fnum:fnum + size_batch])
        loss.append(temp_loss.cpu().detach().numpy())
    y_pred = convert(model, x)
    cm = confusion_matrix(y, y_pred)
    correct = np.trace(cm)
    loss = np.array(loss)
    mean_loss = loss.mean()
    print('%s: Accuracy %d/%d (%.0f%%), Loss %.3f;' % (name,
                                                       correct, len(y), 100. * correct / len(y), mean_loss))
    return


def train_loop(model, train_x, train_y, test_x, test_y, max_epochs=50):
    dummies_train_y = pd.get_dummies(train_y)
    for epoch in range(max_epochs + 1):
        t = time.time()
        size_batch = 1024
        for fnum in range(0, len(dummies_train_y), size_batch):
            model.train(train_x[fnum:fnum + size_batch], dummies_train_y[fnum:fnum + size_batch])
        elapsed = time.time() - t
        if epoch % 10 == 0:
            print('\nEpoch %d, (%.2f sec per epoch)' % (epoch, elapsed))
            accuary(model, train_x, train_y, 'Train')
            accuary(model, test_x, test_y, 'Test')
    PlotConfusionMatrix(model, train_x, train_y, test_x, test_y)
    return


device = "cpu"
# device = "cuda:0"  # uncomment for running on gpu
model = Model('Handwritten Digits', device)

train_x, train_labels, test_x, test_labels = read_data()
train_loop(model, train_x, train_labels, test_x, test_labels)
ShowExamples(model, test_x, test_labels, 10)
