import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import colorsys

# Make the images 3 dimensional
def add_color(image):
    image = torch.stack([image, image, image], dim=1)
    image = torch.squeeze(image, 0)
    return image

# Give color to the image based on the label
def colorize(image, label, color_dict):
    color = random.randrange(0, len(color_dict[label]))
    for i in range(3):
        image[i][image[i] != 0] *= color_dict[(label)][color][i]/255
            
    tfms = torchvision.transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])
    image = tfms(image)
    
    return image, color
    
# Give color to the image based on the label
def colorize_gaussian(image, label, color_dict):
    hue = int(np.random.normal(color_dict[label], color_dict[10], 1)[0]) % 360 if color_dict[10] != 0 else (color_dict[label]) % 360
    rgb = colorsys.hsv_to_rgb(hue/360, 1, 1)
    for i in range(3):
        image[i][image[i] != 0] *= int(rgb[i] * 255)/255
            
    tfms = torchvision.transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])
    image = tfms(image)
    
    return image, hue
        
# Calculate amount correct and loss of a batch
def calculate_correct_loss(model, loss_fn, images, labels, model_type=2):
    # Add color to each image
    
    # for i in range(len(images)):
    #     if 10 in color_dict:
    #         colorize_gaussian(images[i], labels[i].item(), color_dict)
    #     elif len(color_dict) == 0:
    #         tfms = torchvision.transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])
    #         images[i] = tfms(images[i])
    #     else:
    #         colorize(images[i], labels[i].item(), color_dict)

    # Add extra dimension for the network
    if model_type == 3:
        images = images.unsqueeze(1)
            
    # Put images and labels on gpu
    images = images.cuda()
    labels = labels.cuda()

    # Predicted labels
    preds = model(images)
    # Loss
    loss = loss_fn(preds, labels)
    
    # Top predictions per image
    _, top_preds = torch.max(preds, 1)

    # Predictions and labels back on cpu
    top_preds = top_preds.cpu()
    labels = labels.cpu()

    # Calculate number correct
    predictions = [top_preds[i] == labels[i] for i in range(len(top_preds))]
    correct = np.sum(predictions)
        
    return correct, loss