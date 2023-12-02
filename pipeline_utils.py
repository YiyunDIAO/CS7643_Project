from matplotlib import pyplot as plt
from PIL import Image
import torchvision
import numpy as np
import shutil
import random
import torch
import os

def read_data(pct_test = 0.2, root_dir = 'default'):
    """
    This function is used to move certain pct of pictures randomly to training folder and another pct of pictures to testing folder

    :param pct_test: Pct of Pictures randomly picked as testing picture
    :param root_dir: The root directory to pull pictures from. If it's default, it will be constructed as
    current working directory + /COVID-19_Radiography_Dataset
    :return: nothing will be returned. But following two folders will be constructed:
    ../COVID-19_Radiography_Dataset/train/COVID(or Normal or Viral_Penumonia or Lung_Opacity)
    ../COVID-19_Radiography_Dataset/test/COVID(or Normal or Viral_Penumonia or Lung_Opacity)
    """


    class_names = ['Normal', 'Viral_Pneumonia', 'COVID', 'Lung_Opacity']
    if root_dir == 'default':
        root_dir = os.getcwd() + '/COVID-19_Radiography_Dataset'
    else:
        print('manipulating input images in {}'.format(root_dir))

    # the first if condition serves as a simple sanity check on the root path
    if os.path.isdir(os.path.join(root_dir, class_names[1])):
        # first, create a separate folder just to store train and test dataset
        if os.path.isdir(os.path.join(root_dir, 'test')):
            print('test dir already exists, remove current folder and make a new one')
            shutil.rmtree(os.path.join(root_dir, 'test'))
            os.mkdir(os.path.join(root_dir, 'test'))
        else:
            os.mkdir(os.path.join(root_dir, 'test'))

        if os.path.isdir(os.path.join(root_dir, 'train')):
            print('train dir already exists, remove current folder and make a new one')
            shutil.rmtree(os.path.join(root_dir, 'train'))
            os.mkdir(os.path.join(root_dir, 'train'))
        else:
            os.mkdir(os.path.join(root_dir, 'train'))

        # create sub-folder for each class under the main train-test folder
        for c in class_names:
            os.mkdir(os.path.join(root_dir, 'test', c))
            os.mkdir(os.path.join(root_dir, 'train', c))

        # now, randomly select certain images to train and test
        for c in class_names:
            # iterate and get all images
            images = [x for x in os.listdir(os.path.join(root_dir, c, 'images')) if x.lower().endswith('png')]
            # randomly sample certain amount of images to put into test folder
            num_image = len(os.listdir(os.path.join(root_dir, c, 'images')))
            test_sample = int(pct_test * num_image)
            selected_images = random.sample(images, test_sample)
            print('moving {} samples into test folder and {} into train folder for class {}'.format(test_sample,  num_image-test_sample, c))
            for image in images:
                source_path = os.path.join(root_dir, c, 'images', image)
                if image in selected_images:
                    test_path = os.path.join(root_dir, 'test', c, image)
                    shutil.copy(source_path, test_path)
                else:
                    train_path = os.path.join(root_dir, 'train', c, image)
                    shutil.copy(source_path, train_path)
    else:
        print('path {} does not exist, remember to add underline to Viral Penumonia'.format(os.path.join(root_dir, class_names[1])))

def show_images(images, labels, preds, dataset):
    """
    This is the helper function to show images with predicted labels

    :param images: actual image png
    :param labels: correct label for that image
    :param preds: predicted label for the image
    :param dataset: instance of class ChestXRayDataset, this is used to extract class labels
    :return: None. But images will be shown
    """
    plt.figure(figsize=(16, 9))
    class_names = dataset.class_names
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'

        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

class ChestXRayDataset(torch.utils.data.Dataset):
    """
    This is the class that PyTorch needs to be defined as input to the DataLoader
    """
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images

        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']

        for c in self.class_names:
            self.images[c] = get_images(c)

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])

    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)


def show_preds(model, dl_test, dataset):
    """
    show predicted labels together with initial label
    :param model: the trained model to be called
    :param dl_test: DataLoader for test object
    :param dataset: dataset that fed into show_image() function
    :return:
    """
    model.eval()
    images, labels = next(iter(dl_test))
    outputs  = model(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds, dataset)

# This is the normalization that helps Resnet 18 perform better: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
def load_train_test(batch_size = 6):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size = (224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size = (224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dirs = {
        'normal': os.getcwd() + '/COVID-19_Radiography_Dataset/train/Normal/',
        'viral': os.getcwd() + '/COVID-19_Radiography_Dataset/train/Viral_Pneumonia/',
        'covid': os.getcwd() + '/COVID-19_Radiography_Dataset/train/COVID/',
        'opacity': os.getcwd() + '/COVID-19_Radiography_Dataset/train/Lung_Opacity/'
    }

    test_dir = {
        'normal': os.getcwd() + '/COVID-19_Radiography_Dataset/test/Normal/',
        'viral': os.getcwd() + '/COVID-19_Radiography_Dataset/test/Viral_Pneumonia/',
        'covid': os.getcwd() + '/COVID-19_Radiography_Dataset/test/COVID/',
        'opacity': os.getcwd() + '/COVID-19_Radiography_Dataset/test/Lung_Opacity/'
    }

    # First, two instances of ChestXRayDataset needs to be defined
    train_dataset = ChestXRayDataset(train_dirs, train_transform)
    test_dataset = ChestXRayDataset(test_dir, test_transform)
    # Create two DataLoader instances. Training of PyTorch models is based on DataLoader
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataset, test_dataset, dl_train, dl_test
