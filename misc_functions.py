"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import cv2
import numpy as np
print("what?")
import torch
from torch.autograd import Variable
from PyTModel import *
print("hehe")
import torchvision.transforms
from torchvision import models
print("how?")

def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale

    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('results', file_name+ '.jpg')
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)


def save_class_activation_on_image(org_img, activation_map, file_name,actNum):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # Grayscale activation map
    path_to_file = os.path.join('results', file_name+"_"+str(actNum)+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('results', file_name+"_"+str(actNum)+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('results', file_name+"_"+str(actNum)+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #print(cv2im)
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def get_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    #example_list = [['input_images/im0016.jpg', 24],
    #                ['input_images/snake.jpg', 56],
    #                ['input_images/cat_dog.png', 243],
    #                ['input_images/spider.png', 72],
    #                ['input_images/apple.JPEG', 948],
    #                ['input_images/eel.JPEG', 390],
    #                ['input_images/bird.JPEG', 13]]

    example_list = [['input_images/snake.jpg', 56],
                    ['input_images/im0016.jpg', 1],
                    ['input_images/im0016.jpg', 2],
                    ['input_images/im0016.jpg', 3],
                    ['input_images/im0016.jpg', 4],
                    ['input_images/im0016.jpg', 5],
                    ['input_images/im0016.jpg', 6],
                    ['input_images/im0016.jpg', 7],
                    ['input_images/im0016.jpg', 8],
                    ['input_images/im0016.jpg', 9],
                    ['input_images/im0016.jpg', 10],
                    ['input_images/im0016.jpg', 11],
                    ['input_images/im0016.jpg', 12],
                    ['input_images/im0016.jpg', 13],
                    ['input_images/im0016.jpg', 14],
                    ['input_images/im0016.jpg', 15],
                    ['input_images/im0016.jpg', 16],
                    ['input_images/im0016.jpg', 17],
                    ['input_images/im0016.jpg', 18],
                    ['input_images/im0016.jpg', 19],
                    ['input_images/im0016.jpg', 20],
                    ['input_images/im0016.jpg', 21],
                    ['input_images/im0016.jpg', 22],
                    ['input_images/im0016.jpg', 23],
                    ['input_images/im0016.jpg', 24],
                    ['input_images/im0016.jpg', 25],
                    ['input_images/im0016.jpg', 26],
                    ['input_images/im0016.jpg', 27]]
    example_list2 = [['input_images/im0401.jpg', 0],
                    ['input_images/im0401.jpg', 1],
                    ['input_images/im0401.jpg', 2],
                    ['input_images/im0401.jpg', 3],
                    ['input_images/im0401.jpg', 4],
                    ['input_images/im0401.jpg', 5],
                    ['input_images/im0401.jpg', 6],
                    ['input_images/im0401.jpg', 7],
                    ['input_images/im0401.jpg', 8],
                    ['input_images/im0401.jpg', 9],
                    ['input_images/im0401.jpg', 10],
                    ['input_images/im0401.jpg', 11],
                    ['input_images/im0401.jpg', 12],
                    ['input_images/im0401.jpg', 13],
                    ['input_images/im0401.jpg', 14],
                    ['input_images/im0401.jpg', 15],
                    ['input_images/im0401.jpg', 16],
                    ['input_images/im0401.jpg', 17],
                    ['input_images/im0401.jpg', 18],
                    ['input_images/im0401.jpg', 19],
                    ['input_images/im0401.jpg', 20],
                    ['input_images/im0401.jpg', 21],
                    ['input_images/im0401.jpg', 22],
                    ['input_images/im0401.jpg', 23],
                    ['input_images/im0401.jpg', 24],
                    ['input_images/im0401.jpg', 25],
                    ['input_images/im0401.jpg', 26],
                    ['input_images/im0401.jpg', 27]]
    example_list3 = [['input_images/im0811.jpg', 0],
                    ['input_images/im0811.jpg', 1],
                    ['input_images/im0811.jpg', 2],
                    ['input_images/im0811.jpg', 3],
                    ['input_images/im0811.jpg', 4],
                    ['input_images/im0811.jpg', 5],
                    ['input_images/im0811.jpg', 6],
                    ['input_images/im0811.jpg', 7],
                    ['input_images/im0811.jpg', 8],
                    ['input_images/im0811.jpg', 9],
                    ['input_images/im0811.jpg', 10],
                    ['input_images/im0811.jpg', 11],
                    ['input_images/im0811.jpg', 12],
                    ['input_images/im0811.jpg', 13],
                    ['input_images/im0811.jpg', 14],
                    ['input_images/im0811.jpg', 15],
                    ['input_images/im0811.jpg', 16],
                    ['input_images/im0811.jpg', 17],
                    ['input_images/im0811.jpg', 18],
                    ['input_images/im0811.jpg', 19],
                    ['input_images/im0811.jpg', 20],
                    ['input_images/im0811.jpg', 21],
                    ['input_images/im0811.jpg', 22],
                    ['input_images/im0811.jpg', 23],
                    ['input_images/im0811.jpg', 24],
                    ['input_images/im0811.jpg', 25],
                    ['input_images/im0811.jpg', 26],
                    ['input_images/im0811.jpg', 27]]
    example_list4 = [['input_images/im0860.jpg', 0],
                    ['input_images/im0860.jpg', 1],
                    ['input_images/im0860.jpg', 2],
                    ['input_images/im0860.jpg', 3],
                    ['input_images/im0860.jpg', 4],
                    ['input_images/im0860.jpg', 5],
                    ['input_images/im0860.jpg', 6],
                    ['input_images/im0860.jpg', 7],
                    ['input_images/im0860.jpg', 8],
                    ['input_images/im0860.jpg', 9],
                    ['input_images/im0860.jpg', 10],
                    ['input_images/im0860.jpg', 11],
                    ['input_images/im0860.jpg', 12],
                    ['input_images/im0860.jpg', 13],
                    ['input_images/im0860.jpg', 14],
                    ['input_images/im0860.jpg', 15],
                    ['input_images/im0860.jpg', 16],
                    ['input_images/im0860.jpg', 17],
                    ['input_images/im0860.jpg', 18],
                    ['input_images/im0860.jpg', 19],
                    ['input_images/im0860.jpg', 20],
                    ['input_images/im0860.jpg', 21],
                    ['input_images/im0860.jpg', 22],
                    ['input_images/im0860.jpg', 23],
                    ['input_images/im0860.jpg', 24],
                    ['input_images/im0860.jpg', 25],
                    ['input_images/im0860.jpg', 26],
                    ['input_images/im0860.jpg', 27]]
    example_list5 = [['input_images/im0117.jpg', 0],
                     ['input_images/im0117.jpg', 1],
                     ['input_images/im0117.jpg', 2],
                     ['input_images/im0117.jpg', 3],
                     ['input_images/im0117.jpg', 4],
                     ['input_images/im0117.jpg', 5],
                     ['input_images/im0117.jpg', 6],
                     ['input_images/im0117.jpg', 7],
                     ['input_images/im0117.jpg', 8],
                     ['input_images/im0117.jpg', 9],
                     ['input_images/im0117.jpg', 10],
                     ['input_images/im0117.jpg', 11],
                     ['input_images/im0117.jpg', 12],
                     ['input_images/im0117.jpg', 13],
                     ['input_images/im0117.jpg', 14],
                     ['input_images/im0117.jpg', 15],
                     ['input_images/im0117.jpg', 16],
                     ['input_images/im0117.jpg', 17],
                     ['input_images/im0117.jpg', 18],
                     ['input_images/im0117.jpg', 19],
                     ['input_images/im0117.jpg', 20],
                     ['input_images/im0117.jpg', 21],
                     ['input_images/im0117.jpg', 22],
                     ['input_images/im0117.jpg', 23],
                     ['input_images/im0117.jpg', 24],
                     ['input_images/im0117.jpg', 25],
                     ['input_images/im0117.jpg', 26],
                     ['input_images/im0117.jpg', 27]]
    selected_example = example_index
    img_path = example_list[selected_example][0]
    target_class = example_list[selected_example][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    #print(img_path)
    original_image = cv2.imread(img_path, 1)
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model

    pretrained_model = models.vgg19(pretrained=True)#Net()
    #mod = list(pretrained_model.classifier.children())
    #mod.pop()
    #mod.append(torch.nn.Linear(4096, 28))
    #new_classifier = torch.nn.Sequential(*mod)
    #pretrained_model.classifier = new_classifier

    #pretrained_model.load_state_dict(torch.load('net.pth'))  # models.vgg19(pretrained=True)
    #pretrained_model.load_state_dict(torch.load('net.pth', map_location=lambda storage, loc: storage))
    #print(pretrained_model)
    #pretrained_model= models.vgg19(pretrained=True)
    #pretrained_model = models.resnet18(pretrained=True)
    #print(pretrained_model)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)