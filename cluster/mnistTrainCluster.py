import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from small2DNet import small2DNet
from small3DNet import small3DNet
from util import add_color, colorize, colorize_gaussian, calculate_correct_loss
from colorMNist import colorMNist
import pickle
import sys
import os

def create_dirs(path, dataset_sizes):
    for x in dataset_sizes:
        for iteration in range(2, 13, 2):
            os.makedirs(path + str(x) + '/' + str(iteration))
    
# Train 6 models for a model type, dataset size and dataset std
def trainNet(dataset_std, dataset_size, model_type, model_layers2, model_layers3, features, n_exp, num_epochs, freeze_weights):    
    dataset_filename = ""
    if dataset_std == 0:
        dataset_filename = "cmnist_deterministic"
    elif dataset_std == 1000000:
        dataset_filename = "cmnist_gaussian_uniform"
    else:
        dataset_filename = "cmnist_gaussian_" + str(dataset_std)

    sfile_fill = str(dataset_std) if dataset_std != 1000000 else "uniform"

    # Load data from pickle file
    with open("/tudelft.net/staff-umbrella/StudentsCVlab/itahur/custom_datasets/" + str(dataset_size) + "/" + dataset_filename + ".pkl", "rb") as f:
        cmnist_train, cmnist_val, cmnist_test = pickle.load(f)

        # Create datasets
        train_dataset = colorMNist(cmnist_train)
        val_dataset = colorMNist(cmnist_val)
        test_dataset = colorMNist(cmnist_test)

        # Dataloaders
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 0)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers = 0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 0)

        for iteration in tqdm(range(2, 13, 2), total=6, desc=f"ds std: {dataset_std}, ds size: {dataset_size}"):
            # print("Seed:", iteration)
            # Model layers
            model_layers = model_layers2 if model_type == 2 else model_layers3
            # Set seed
            torch.manual_seed(iteration)
            # Create model
            model = small2DNet(model_layers, model_layers[-2], features[0]) if model_type == 2 else small3DNet(model_layers, model_layers[-2], features[1])
            # Load file and save file
            # lfile = "Gaussian3D_" + sfile_fill
            sfile = "Gaussian" + str(model_type) + "D_" + sfile_fill
            # Load model
            # model.load_state_dict(torch.load('model_saves/' + str(iteration) + '/' + str(ds_size) + '/cmnist/'+ lfile + '.pth'))
            # Put model on gpu
            model.cuda()
            # Loss function
            loss_fn = torch.nn.CrossEntropyLoss()
            # Optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

            # Freeze weights
            if freeze_weights:
                children = [x for x in model.children()]
                for x in children[0]:
                    for param in x.parameters():
                        param.requires_grad = False

            # Number of epochs to train
            epochs = num_epochs

            # Placeholder variables to put training and validation accuracies and losses per epoch
            train_accuracies = []
            train_losses = []
            val_accuracies = []
            val_losses = []

            # for epoch in tqdm(range(epochs), total=epochs, desc='Training'):
            for epoch in range(epochs):

                # Put model on training mode
                model.train()
                train_total_correct = 0
                train_total_loss = []

                for (images, labels) in train_dataloader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Calculate number correct and loss in batch
                    correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type=model_type)

                    # Backpropagation
                    loss.backward()
                    # Step function
                    optimizer.step()

                    # Update amount correct and loss with current batch
                    train_total_correct += correct
                    train_total_loss.append(loss.item())

                # Append epoch accuracy and loss
                train_accuracies.append(train_total_correct / len(train_dataset))
                train_losses.append(sum(train_total_loss) / len(train_total_loss))

                # Put model on evaluation mode
                model.eval()
                val_total_correct = 0
                val_total_loss = []

                # Without gradient calculation
                with torch.no_grad():
                    for (images, labels) in val_dataloader:

                        # Calculate number correct and loss in batch
                        correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type=model_type)

                        # Update amount correct and loss with current batch
                        val_total_correct += correct
                        val_total_loss.append(loss.item())

                # Append epoch accuracy and loss
                val_accuracies.append(val_total_correct / len(val_dataset))
                val_losses.append(sum(val_total_loss) / len(val_total_loss))

            # Save training and validation accuracies and losses
            with open('Experiments/cmnist/trainvalAccs/' + str(n_exp) + '/' + str(dataset_size) + '/' + str(iteration) + '/' + sfile + '.txt', 'w') as f:
                for i in range(epochs):
                    f.write("Epoch " + str(i + 1) + "\n")
                    f.write("Train acc and loss\t" + str(train_accuracies[i]) + "\t" + str(train_losses[i]) + "\n")
                    f.write("Val acc and loss\t" + str(val_accuracies[i]) + "\t" + str(val_losses[i]) + "\n")

            # Save the model
            torch.save(model.state_dict(), 'Experiments/cmnist/model_saves/' + str(n_exp) + '/' + str(dataset_size) + '/' + str(iteration) + '/' + sfile + '.pth')
            
def trainNets(dataset_stds, dataset_sizes, model_type, model_layers2, model_layers3, features, n_exp, num_epochs=20, freeze_weights=False):
    if not os.path.exists('./Experiments/cmnist/model_saves/' + str(n_exp)):
        create_dirs('./Experiments/cmnist/model_saves/' + str(n_exp) + '/', dataset_sizes)
    if not os.path.exists('./Experiments/cmnist/trainvalAccs/' + str(n_exp)):
        create_dirs('./Experiments/cmnist/trainvalAccs/' + str(n_exp) + '/', dataset_sizes)
        
    for size in dataset_sizes:
        for std in dataset_stds:
            trainNet(std, size, model_type, model_layers2, model_layers3, features, n_exp, num_epochs, freeze_weights)