
from monai.utils import first, set_determinism
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset

import torch
import matplotlib.pyplot as plt

import os
from glob import glob
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from monai.inferers import sliding_window_inference

def testing(data):
    data_train = {'brain': 'dataset3d/Task01_BrainTumour/Data_Train_Test', 'heart': 'dataset3d/Task02_Heart/Data_Train_Test', 'liver': 'dataset3d/Task03_Liver/Data_Train_Test'}

    if data=="brain":
        model_dir = 'resultsbrain' 
    elif data=="heart":
        model_dir = 'resultsheart' 
    else:
        model_dir = 'resultsliver' 

    in_dir = data_train[data.lower()]

    train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
    train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
    test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
    test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

    plt.figure("Results 25 june", (12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Train dice loss")
    x = [i + 1 for i in range(len(train_loss))]
    y = train_loss
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 2)
    plt.title("Train metric DICE")
    x = [i + 1 for i in range(len(train_metric))]
    y = train_metric
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 3)
    plt.title("Test dice loss")
    x = [i + 1 for i in range(len(test_loss))]
    y = test_loss
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 4)
    plt.title("Test metric DICE")
    x = [i + 1 for i in range(len(test_metric))]
    y = test_metric
    plt.xlabel("epoch")
    plt.plot(x, y)

    # plt.show()

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]
    test_files = test_files[0:9]

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),   
            ToTensord(keys=["vol", "seg"]),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

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

    model.load_state_dict(torch.load(
        os.path.join(model_dir, "best_metric_model.pth")))
    model.eval()

    sw_batch_size = 4
    roi_size = (128, 128, 64)
    with torch.no_grad():
        test_patient = first(test_loader)
        t_volume = test_patient['vol']
        #t_segmentation = test_patient['seg']
        
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53
            
        for i in range(32):
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_patient["seg"][0, 0, :, :, i] != 0)
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
        plt.savefig('media/' + data + '/output.png')
        plt.imsave('media/' + data + '/output1.png', test_patient["vol"][0, 0, :, :, i], cmap="gray")
        plt.imsave('media/' + data + '/output2.png', test_patient["seg"][0, 0, :, :, i] != 0)
        plt.imsave('media/' + data + '/output3.png', test_outputs.detach().cpu()[0, 1, :, :, i])


# data = input("Enter the desired testing: ")
# testing(data)