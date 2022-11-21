import numpy as np
import torch
from small2DNet import small2DNet
from small3DNet import small3DNet
from mnistTrainCluster import trainNets
import sys

if __name__ == '__main__':
    dataset_stds = [0, 12, 60, 120, 1000000]
    dataset_sizes = ["1k", "2k", "5k", "10k", "20k", "60k"]
    model_type = int(sys.argv[1])
    n_exp = int(sys.argv[2])
    print("model_type: " + str(model_type) + ", n_exp: " + str(n_exp))
    model_layers2 = []
    model_layers3 = []
    features = []
    num_epochs = 20
    
    # 3D always at least has a feature in each color channel
    # Equal features, 2D more spatial
    if n_exp == 1:
        model_layers2 = [16, "M", 48, "M", 64, "M"]
        model_layers3 = [16, "M", 16, "M", 64, "M"]
        features = [[3, 1], [3, 1, 1]]
    # Equal features, 2D equal spatial
    elif n_exp == 2:
        model_layers2 = [16, "M", 48, "M", 64, "M"]
        model_layers3 = [16, "M", 16, "M", 64, "M"]
        features = [[1, 1], [3, 1, 1]]
    # Equal features, 2D more spatial, more linear neurons
    elif n_exp == 3:
        model_layers2 = [16, "M", 48, "M", 80, "M"]
        model_layers3 = [16, "M", 16, "M", 80, "M"]
        features = [[3, 1], [3, 1, 1]]
    # Equal features, 2D equal spatial, more linear neurons
    elif n_exp == 4:
        model_layers2 = [16, "M", 48, "M", 80, "M"]
        model_layers3 = [16, "M", 16, "M", 80, "M"]
        features = [[1, 1], [3, 1, 1]]
    elif n_exp == 5:
        dataset_stds = [120]
        dataset_sizes = ["5k"]
        num_epochs = 50
        model_layers2 = [16, "M", 48, "M", 64, "M"]
        model_layers3 = [16, "M", 16, "M", 64, "M"]
        features = [[3, 1], [3, 1, 1]]
    # Equal features, 2D equal spatial, collapsed 3D
    elif n_exp == 6:
        model_layers2 = [16, "M", 48, "M", 64, "M"]
        model_layers3 = [16, "M", 16, "M", 64, "M"]
        features = [[1, 1], [1, 1, 1]]
    # Equal features, 2D equal spatial
    elif n_exp == 7:
        model_layers2 = [28, "M", 55, "M", 111, "M"]
        model_layers3 = [16, "M", 32, "M", 64, "M"]
        features = [[1, 1], [3, 1, 1]]
    # 7 but more neurons
    elif n_exp == 8:
        model_layers2 = [56, "M", 110, "M", 222, "M"]
        model_layers3 = [32, "M", 64, "M", 128, "M"]
        features = [[1, 1], [3, 1, 1]]
    # 8 but more neurons
    elif n_exp == 9:
        model_layers2 = [84, "M", 165, "M", 333, "M"]
        model_layers3 = [48, "M", 96, "M", 192, "M"]
        features = [[1, 1], [3, 1, 1]]
    # 7 but increase last layer
    elif n_exp == 10:
        model_layers2 = [28, "M", 55, "M", 222, "M"]
        model_layers3 = [16, "M", 32, "M", 128, "M"]
        features = [[1, 1], [3, 1, 1]]
    # 10 but increase last layer
    elif n_exp == 11:
        model_layers2 = [28, "M", 55, "M", 333, "M"]
        model_layers3 = [16, "M", 32, "M", 192, "M"]
        features = [[1, 1], [3, 1, 1]]
    # 8 but more neurons
    elif n_exp == 12:
        model_layers2 = [112, "M", 220, "M", 444, "M"]
        model_layers3 = [48, "M", 96, "M", 192, "M"]
        features = [[1, 1], [3, 1, 1]]
    # best configurations 2D and 3D, collapsed AAP
    elif n_exp == 13:
        model_layers2 = [84, "M", 165, "M", 333, "M"]
        model_layers3 = [32, "M", 64, "M", 128, "M"]
        features = [[1, 1], [1, 1, 1]]
    # best configurations 2D and 3D, non-collapsed AAP
    elif n_exp == 14:
        model_layers2 = [84, "M", 165, "M", 333, "M"]
        model_layers3 = [32, "M", 64, "M", 128, "M"]
        features = [[1, 1], [3, 1, 1]]
    # best configurations 2D and 3D, collapsed AAP
    elif n_exp == 15:
        model_layers2 = [84, "M", 165, "M", 333, "M"]
        model_layers3 = [32, "M", 64, "M", 128, "M"]
        features = [[1, 1], [1, 1, 1]]
    # fair configurations 2D and 3D, non-collapsed AAP
    elif n_exp == 16:
        model_layers2 = [28, "M", 55, "M", 111, "M"]
        model_layers3 = [16, "M", 32, "M", 64, "M"]
        features = [[1, 1], [3, 1, 1]]
    # fair configurations 2D and 3D, collapsed AAP
    elif n_exp == 16:
        model_layers2 = [28, "M", 55, "M", 111, "M"]
        model_layers3 = [16, "M", 32, "M", 64, "M"]
        features = [[1, 1], [1, 1, 1]]
    # Not equal increase, 2D more increase but equal spatial features
    
    trainNets(dataset_stds, dataset_sizes, model_type, model_layers2, model_layers3, features, n_exp, num_epochs=num_epochs)