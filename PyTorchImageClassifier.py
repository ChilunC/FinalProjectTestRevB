
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
import matplotlib.pyplot as plt
import numpy as np

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

#import dataset from .csv file referencing the image and the points for the joints in the image
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

#transform image to fit the VGG19 model
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




# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#get training dataset
face_dataset_train = FaceLandmarksDataset(csv_file='jointsTrain.csv',
                                           root_dir='fulljoints/',
                                           transform=data_transforms['train'] #transforms.Compose([
                                               #transform.Rescale(32),
                                               #RandomCrop(32),
                                               #ToTensor()
                                           )
#get test dataset
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


#print(trainloader)
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

net = NetFeat()


#import a pretrained VGG19 model and replace the last layer with a custom layer
#net = torchvision.models.vgg19(pretrained=True)
#for param in net.features.parameters():
#    param.requires_grad = False
#for param in net.classifier.parameters():
#    param.requires_grad=False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
#mod = list(net.classifier.children())
#mod.pop()

#mod.append(torch.nn.Linear(4096,28))
#new_classifier = torch.nn.Sequential(*mod)
#net.classifier = new_classifier
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

optimizer = optim.SGD(net.classifier[-1].parameters(), lr=0.0001, momentum=0.9)

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
torch.save(net.state_dict(),'netLun.pth')

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
#for data in testloader:
for i, data in enumerate(testloader, 0):
    images, labels = data
    target_var = torch.autograd.Variable(labels,volatile=True)
    input_var = torch.autograd.Variable(images,volatile=True)

    #optimizer.zero_grad()
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
