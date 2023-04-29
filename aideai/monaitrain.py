from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preporcess import prepare
from utilities import train


def training(data):
    data_train = {'brain': 'dataset3d/Task01_BrainTumour/Data_Train_Test', 'heart': 'dataset3d/Task02_Heart/Data_Train_Test', 'liver': 'dataset3d/Task03_Liver/Data_Train_Test'}

    if data=="brain":
        model_dir = 'resultsbrain' 
    elif data=="heart":
        model_dir = 'resultsheart' 
    else:
        model_dir = 'resultsliver' 

    data_dir = data_train[data.lower()]

    data_in = prepare(data_dir, cache=True)
    print("data_in: ", data_in)
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)


    #loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
    loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
    train(model, data_in, loss_function, optimizer, 60, model_dir)

if __name__ == '__main__':
    data = input("Enter the desired training: ")
    training(data)
    