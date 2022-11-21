import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from small2DNet import small2DNet
from small3DNet import small3DNet
from util import add_color, colorize, colorize_gaussian, calculate_correct_loss
from colorMNist import colorMNist
import pickle

for ds in [1000000]:
    print(ds)
    textfill = ""
    if ds == 0:
        textfill = "cmnist_deterministic"
    elif ds == 1000000:
        textfill = "cmnist_gaussian_uniform"
    else:
        textfill = "cmnist_gaussian_" + str(ds)
    
    for hs in ["90", "180", "270"]:
        print(hs)
        textfill = "cmnist_deterministic"
        if hs != "0":
            textfill = textfill + "_" + hs

        sfile_fill = str(ds) if ds != 1000000 else "uniform"

        ds_size = "5k"

        # Load data from pickle file
        with open("/tudelft.net/staff-umbrella/StudentsCVlab/itahur/custom_datasets/" + str(ds_size) + "/" + textfill + ".pkl", "rb") as f:
            cmnist_train, cmnist_val, cmnist_test = pickle.load(f)

            # Create datasets
            train_dataset = colorMNist(cmnist_train)
            val_dataset = colorMNist(cmnist_val)
            test_dataset = colorMNist(cmnist_test)

            # Dataloaders
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 0)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers = 0)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 0)

            for iteration in tqdm(range(10, 13, 12), total=6, desc="Seed"):
                # print("Seed:", iteration)
                # 2D layers
                model_layers2 = [84, "M", 165, "M", 333, "M"]
                # 3D layers
                model_layers3 = [32, "M", 64, "M", 128, "M"]
                features = [[1, 1], [3, 1, 1]]
                # Set seed
                torch.manual_seed(iteration)
                # Create model
                model = small3DNet(model_layers3, model_layers3[-2], features[1])
                mtype = len(model.features[0].kernel_size)
                # Load file and save file
                lfile = "Gaussian" + str(mtype) + "D_" + sfile_fill
                sfile = "D-Gaussian" + str(mtype) + "D_" + sfile_fill + "_" + hs
                # Load model
                model.load_state_dict(torch.load('Experiments/cmnist/model_saves/8/' + str(ds_size) + '/' + str(iteration) + '/'+ lfile + '.pth'))
                # Put model on gpu
                model.cuda()
                # Loss function
                loss_fn = torch.nn.CrossEntropyLoss()
                # Optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

                # Freeze weights
                children = [x for x in model.children()]
                for x in children[0]:
                    for param in x.parameters():
                        param.requires_grad = False

                # Number of epochs to train
                epochs = 20

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
                        correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type=mtype)

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
                            correct, loss = calculate_correct_loss(model, loss_fn, images, labels, model_type=mtype)

                            # Update amount correct and loss with current batch
                            val_total_correct += correct
                            val_total_loss.append(loss.item())

                    # Append epoch accuracy and loss
                    val_accuracies.append(val_total_correct / len(val_dataset))
                    val_losses.append(sum(val_total_loss) / len(val_total_loss))

                with open('Experiments/cmnist/trainvalAccs/best/' + str(ds_size) + '/' + str(iteration) + '/' + sfile + '.txt', 'w') as f:
                    for i in range(epochs):
                        f.write("Epoch " + str(i + 1) + "\n")
                        f.write("Train acc and loss\t" + str(train_accuracies[i]) + "\t" + str(train_losses[i]) + "\n")
                        f.write("Val acc and loss\t" + str(val_accuracies[i]) + "\t" + str(val_losses[i]) + "\n")

                # Save the model
                torch.save(model.state_dict(), 'Experiments/cmnist/model_saves/best/' + str(ds_size) + '/' + str(iteration) + '/' + sfile + '.pth')