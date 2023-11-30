# link: https://github.com/Hevenicio/Detecting-COVID-19-with-Chest-X-Ray-using-PyTorch/blob/master/Detecting%20COVID-19%20Notebook.ipynb

from matplotlib import pyplot as plt
from PIL import Image
import torchvision
import numpy as np
import shutil
import random
import torch
import os
from pipeline_utils import read_data, ChestXRayDataset, show_images, show_preds, load_train_test
torch.manual_seed(0)

print('Using PyTorch version', torch.__version__)

###########################################
# Data Prep step
###########################################
# the function below is used to create the desired folder structure
# May only need to run this function once
read_data()

train_dataset, test_dataset, dl_train, dl_test = load_train_test()
print('Number of training batches', len(dl_train))
print('Number of testing batches', len(dl_test))

images, labels = next(iter(dl_train))
show_images(images, labels, labels, train_dataset)

###########################################
# modeling part, change your models
###########################################
resnet18 = torchvision.models.resnet18(pretrained = True)

print(resnet18)

resnet18.fc = torch.nn.Linear(in_features = 512, out_features = 3)
loss_fn     = torch.nn.CrossEntropyLoss()
optimizer   = torch.optim.Adam(resnet18.parameters(), lr = 3e-5)

# change your models here
model = resnet18


###########################################
# Below is the pre-defined training function
###########################################
def train(epochs, model, early_stopping_accuracy=0.8):
    """
    Below are some notes on common parameters
    :param epochs: one full cycle through the training dataset. A cycle is composed of many iterations
    :param model: the model used to train the data
    Other parameters that might be related:
    batch size: the number of training samples used in 1 iteration
    Iterations: the number of batches needed to complete one Epoch
    Num of steps per Epoch = Total number of training samples / batch size

    :return: None, however, the model itself should be updated
    """

    print('Starting training..')
    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.

        model.train() # set model to training phase

        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 20 == 0:
                print('Evaluating at step', train_step)

                accuracy = 0

                model.eval() # set model to eval phase

                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())

                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Testing set Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                show_preds(model, dl_test, test_dataset)

                model.train()
                # set the threshold for early stopping
                if accuracy >= early_stopping_accuracy:
                    print('Performance condition satisfied, stopping..')
                    return

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')

train(epochs=1, model = resnet18)

show_preds(model, dl_test, test_dataset)
