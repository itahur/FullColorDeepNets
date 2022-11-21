import numpy as np
from convTrainCluster import trainNets
import sys

if __name__ == '__main__':
    dataset_stds = [0, 12, 60, 120, 1000000]
    dataset_sizes = ["1k", "2k", "5k", "10k", "20k", "60k"]
    model_type = int(sys.argv[1])
    n_exp = int(sys.argv[2])
    print("model_type: " + str(model_type) + ", n_exp: " + str(n_exp))
    model_layers1 = []
    model_layers2 = []
    features = []
    num_epochs = 20
    mix = False
    
    if n_exp == 1:
        model_layers1 = [32, "M", 64, "M", 128, "M"]
        model_layers2 = [32, "M", 64, "M", 128, "M"]
        features = [3, 1, 1]
    elif n_exp == 2:
        mix = True
        model_layers1 = [32, "M", 64, "M", 128, "M"]
        model_layers2 = [32, "M", 64, "M", 128, "M"]
        features = [3, 1, 1]
    elif n_exp == 3:
        model_layers1 = [32, "M", 64, "M", 128, "M"]
        model_layers2 = [32, "M", 64, "M", 128, "M"]
        features = [1, 1, 1]
    elif n_exp == 4:
        mix = True
        model_layers1 = [32, "M", 64, "M", 128, "M"]
        model_layers2 = [32, "M", 64, "M", 128, "M"]
        features = [1, 1, 1]
    
    trainNets(dataset_stds, dataset_sizes, model_type, model_layers1, model_layers2, features, n_exp, num_epochs=num_epochs, mix=mix)