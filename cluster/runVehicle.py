from vehicleCluster import train_test
import sys

if __name__ == '__main__':
    print("model_type, n_exp")
    model_type = int(sys.argv[1])
    lr_schedule = []
    model_layers2 = [96, 256, 512, 512, 512]
    model_layers3 = []
    n_epochs = 70
    n_exp = int(sys.argv[2])
    
    if n_exp == 1:
        model_layers3 = [96, 96, 296, 296, 296]
    elif n_exp == 2:
        model_layers3 = [96, 96, 296, 296, 170]
    elif n_exp == 3:
        model_layers3 = [55, 148, 296, 296, 296]
    elif n_exp == 4:
        model_layers3 = [55, 148, 296, 296, 296]
    elif n_exp == 5:
        model_layers3 = [55, 148, 296, 296, 296]
    
    train_test(model_type, n_epochs, model_layers2, model_layers3, lr_schedule, n_exp)