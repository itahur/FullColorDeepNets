import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from PIL import  Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
import glob
from vggm2DNet import vggm2DNet
from vggm3DNet import vggm3DNet
from tqdm import tqdm
from util import calculate_correct_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import sys

class VehicleColorDataset(Dataset):
    def __init__(self, image_list, class_list, transforms = None):
        self.transform = transforms
        self.image_list = image_list
        self.class_list = class_list
        self.data_len = len(self.image_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.class_list[index]

def decode_label(index, labels):
    return labels[index]

def encode_label_from_path(path, labels):
    for index,value in enumerate(labels):
        if value in path:
            return index
        
def train_test(model_type, n_epochs, model_layers2, model_layers3, lr_schedule, n_exp):
    if not os.path.exists('./Experiments/vehicle/model_saves/' + str(n_exp)):
        os.makedirs('./Experiments/vehicle/model_saves/' + str(n_exp))
    if not os.path.exists('./Experiments/vehicle/plots/' + str(n_exp)):
        os.makedirs('./Experiments/vehicle/plots/' + str(n_exp))
        
    labels = ['black', 'blue' , 'cyan' , 'gray' , 'green' , 'red' , 'white' , 'yellow']
    path = '/tudelft.net/staff-umbrella/StudentsCVlab/itahur/color/'
    image_list = glob.glob(path + '**/*')
    class_list = [encode_label_from_path(item, labels) for item in image_list]
    x_trainval, x_test, y_trainval, y_test = train_test_split(image_list, class_list,
                                                            train_size= 0.9, stratify=class_list, shuffle=True, random_state=42)
    class_listtv = [encode_label_from_path(item, labels) for item in x_trainval]
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, class_listtv,
                                                            train_size= 0.8, stratify=class_listtv, shuffle=True, random_state=42)

    b_size = 64
    transforms=Compose([Resize(224), CenterCrop(224), ToTensor()])
    train_dataset = VehicleColorDataset(x_train, y_train, transforms)
    train_dataloader = DataLoader(train_dataset, batch_size = b_size)
    val_dataset = VehicleColorDataset(x_val, y_val, transforms)
    val_dataloader = DataLoader(val_dataset, batch_size = b_size)
    test_dataset = VehicleColorDataset(x_test, y_test, transforms)
    test_dataloader = DataLoader(test_dataset, batch_size = b_size)

    # Layers of the model
    model_layers = model_layers2 if model_type == 2 else model_layers3
    # Set seed
    torch.manual_seed(12)
    # Create model
    model = vggm2DNet(model_layers) if model_type == 2 else vggm3DNet(model_layers)
    # Load file and save file
    # lfile = "Gaussian2D_12"
    sfile = "Vehicle" + str(model_type) + "D"
    # Load model
    # model.load_state_dict(torch.load('model_saves/new_fair/'+ lfile + '.pth'))
    # Put model on gpu
    model.cuda()
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dataloader), epochs=n_epochs)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    epochs = n_epochs
    for epoch in range(epochs):
                
        with tqdm(train_dataloader, unit="batch") as tepoch:
            model.train()
            train_epoch_loss = []
            train_epoch_correct = 0

            for images, labels in tepoch:
                tepoch.set_description(f"Train | Epoch {epoch}")
                optimizer.zero_grad()
                correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type)

                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_epoch_loss.append(loss.detach().item())
                train_epoch_correct += correct

                tepoch.set_postfix(loss = sum(train_epoch_loss) / len(train_epoch_loss),
                                   accuracy = train_epoch_correct / (len(train_epoch_loss) * b_size + len(images)) * 100)

            train_losses.append(sum(train_epoch_loss) / len(train_epoch_loss))
            train_accs.append(train_epoch_correct / len(train_dataset))

        with torch.no_grad():
            model.eval()
            with tqdm(val_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Val | Epoch {epoch}")
                val_epoch_loss = []
                val_epoch_correct = 0

                for images, labels in tepoch:
                    correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type)

                    val_epoch_loss.append(loss.detach().item())
                    val_epoch_correct += correct

                    tepoch.set_postfix(loss = sum(val_epoch_loss) / len(val_epoch_loss),
                                       accuracy = val_epoch_correct / (len(val_epoch_loss) * b_size + len(images)) * 100)

            val_losses.append(sum(val_epoch_loss) / len(val_epoch_loss))
            val_accs.append(val_epoch_correct / len(val_dataset))

        with torch.no_grad():
            model.eval()
            with tqdm(test_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Test | Epoch {epoch}")
                test_epoch_loss = []
                test_epoch_correct = 0
                for images, labels in tepoch:
                    correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type)

                    test_epoch_loss.append(loss.detach().item())
                    test_epoch_correct += correct

                    tepoch.set_postfix(loss = sum(test_epoch_loss) / len(test_epoch_loss),
                                           accuracy = test_epoch_correct / (len(test_epoch_loss) * b_size + len(images)) * 100)

                test_losses.append(sum(test_epoch_loss) / len(test_epoch_loss))
                test_accs.append(test_epoch_correct / len(test_dataset))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'Experiments/vehicle/model_saves/' + str(n_exp) + '/VGGM_' + str(epoch + 1) + 'E' + sfile + '.pth')
            np.savetxt('Experiments/vehicle/plots/' + str(n_exp) + '/VGGM_' + str(epoch + 1) + 'E' + sfile + '.txt',
                       [train_losses, train_accs, val_losses, val_accs, test_losses, test_accs])