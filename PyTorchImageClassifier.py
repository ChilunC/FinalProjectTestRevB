# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.

Specifically for ``vision``, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import pandas as pd
import os
from skimage import io, transform
import PIL
from PIL import Image
from PyTModel import *

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image),
                torch.from_numpy(landmarks))


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]+'.jpg')
        #print(img_name)
        #image = io.imread(img_name)
        image = Image.open(img_name)
        image = np.array(image)
        #image = np.rollaxis(image,2,0)

        #image.astype(int)
        landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
        #landmarks = landmarks.reshape(-1, 2)
        image = np.array(image).astype(np.float32)
        landmarks = np.array(landmarks).astype(np.int)
        #print("landmarks.shape")
        #print(landmarks.shape)
        #landmarks.astype(int)
        #sample = {'image': image, 'landmarks': landmarks}
        print(image.shape)
        sample = image,landmarks
        if self.transform:
            image = self.transform(image)
        #sample = (image, landmarks)
        print(image.shape)
        return image, landmarks #sample




########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transform = transforms.Compose(
    [transforms.ToTensor()])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)

#trainloaderC = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=True, num_workers=2)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#landmarks_frame = pd.read_csv('joints/joints.csv')
#landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
#landmarks = landmarks.reshape(-1, 2)

#face_dataset_train = FaceLandmarksDataset(csv_file='jointsTrain.csv',
#                                    root_dir='joints/')
#face_dataset_test = FaceLandmarksDataset(csv_file='jointsTest.csv',
#                                    root_dir='joints/')

#trainloader = DataLoader(face_dataset_train, batch_size=4,
#                        shuffle=True, num_workers=2)
#testloader = DataLoader(face_dataset_test, batch_size=4,
#                        shuffle=True, num_workers=2)

#print(landmarks_frame)
# get some random training images
#print(trainloader)

#scale = Rescale(32)
#crop = RandomCrop(128)
#composed = transforms.Compose([Rescale(32),
#                               RandomCrop(224)])

face_dataset_train = FaceLandmarksDataset(csv_file='jointsTrain.csv',
                                           root_dir='fulljoints/',
                                           transform=data_transforms['train'] #transforms.Compose([
                                               #transform.Rescale(32),
                                               #RandomCrop(32),
                                               #ToTensor()
                                           )
#print("Facedataset")
#print(face_dataset_train)
face_dataset_test = FaceLandmarksDataset(csv_file='jointsTest.csv',
                                           root_dir='fulljoints/',
                                           transform=data_transforms['val'] #.Compose([
                                               #Rescale(32),
                                               #RandomCrop(32),
                                               #ToTensor()
                                           )
trainloader = DataLoader(face_dataset_train, batch_size=4,
                        shuffle=True, num_workers=2)
testloader = DataLoader(face_dataset_test, batch_size=4,
                        shuffle=True, num_workers=2)

#dataiter = iter(trainloaderC)
#images, labels = dataiter.next()
#print("image type data")
#print(type(images))
#print(images.shape)
print(trainloader)
dataiter = iter(trainloader)
print(dataiter)
images, labels = dataiter.next()
print(images)
#print("fromiter")
#print(labels)
#print(images)
#print(images['image'])
#print(labels)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F




net = torchvision.models.vgg19(pretrained=True)
for param in net.parameters():
    param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
mod = list(net.classifier.children())
mod.pop()
mod.append(torch.nn.Linear(4096,28))
new_classifier = torch.nn.Sequential(*mod)
net.classifier = new_classifier
if torch.cuda.is_available():
    net.cuda()
#net.features[-1] = nn.Linear(4096, 28) # assuming that the fc7 layer has 512 neurons, otherwise change it
#net = Net()

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()  # this is for regression mean squared loss

optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        #labels = torch.squeeze(labels,1)
        #labels = torch.squeeze(labels, 1)
        #labels = torch.squeeze(labels, 1)
        #print("labels")
        #print(labels.size())
        #print(inputs.size())
        #inputs.type(torch.ByteTensor)
        # wrap them in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        #print("inputs")
        #print(inputs)
        # print(labels)
        #labels = labels[0,:]
        #print("after try")
        #print(labels)
        #print(type(inputs))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs)
        loss = 0
        for o in range(len(outputs)):
            loss += criterion(outputs[o], labels[o].type('torch.FloatTensor'))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(),'net.pth')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

#dataiter = iter(testloader)
#images, labels = dataiter.next()
#print("labels")
#print(type(labels))
# print images
#imshow(torchvision.utils.make_grid(images))
#loss = criterion(outputs, labels.type('torch.FloatTensor'))
#print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

#outputs = net(Variable(images))

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
#_, predicted = torch.max(outputs.data, 1)

#print('Predicted: ', ' '.join(classes[predicted[j]]
#                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
for data in testloader:
    images, labels = data
    target_var = torch.autograd.Variable(labels,volatile=True)
    input_var = torch.autograd.Variable(images,volatile=True)
    outputs = net(input_var)
    score_map = outputs[-1].data
    #_, predicted = torch.max(outputs.data, 1)
    loss = 0
    for o in outputs:
        loss += criterion(o, target_var.type('torch.FloatTensor'))
    #acc = accuracy(score_map,labels)
    total += labels.size(0)
    correct += (loss.data[0]<10)
    print("loss")
    print(loss)
    print("total")
    print(total)
    print("correct")
    print(correct)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor on to the GPU, you transfer the neural
# net onto the GPU.
# This will recursively go over all modules and convert their parameters and
# buffers to CUDA tensors:
#
# .. code:: python
#
#     net.cuda()
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ::
#
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is realllly small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train an face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train an face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: http://pytorch.slack.com/messages/beginner/
