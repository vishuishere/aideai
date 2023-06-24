
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

import cv2
import numpy as np
import os
from glob import glob
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from monai.inferers import sliding_window_inference

import pydicom
from pydicom import dcmread
from pydicom.dataset import Dataset as Dicom_dataset
from pydicom.uid import generate_uid
from PIL import Image

def dicom_conversion(path, image_path, project_name, sample_name, data):
    print(project_name.capitalize(), data.capitalize(), sample_name.capitalize())
    print(path)
    image = Image.open(image_path)

    # Convert the image to grayscale if it has more than one channel
    image = image.convert("L")

    # Create a new DICOM dataset
    dataset = Dicom_dataset()

    # Set the necessary DICOM attributes
    dataset.PatientName = project_name.capitalize()
    dataset.StudyDescription = data.capitalize() +" Segmentation"
    dataset.SeriesDescription = "Sample Name: " + sample_name.capitalize()
    dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
    dataset.SOPInstanceUID = generate_uid()
    dataset.Modality = 'OT'
    dataset.Rows, dataset.Columns = image.size
    dataset.PixelRepresentation = 0  # Unsigned integer
    dataset.SamplesPerPixel = 1  # Grayscale image
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelData = image.tobytes()

    # Set the transfer syntax attributes
    dataset.is_little_endian = True
    dataset.is_implicit_VR = True

    # Save the DICOM dataset to a file
    output_path = path.replace('/output/', '/') + "output.dcm"  # Path to save the converted DICOM file
    dataset.save_as(output_path)

    # Read the saved DICOM file and print its information
    saved_dataset = dcmread(output_path, force=True)
    print(saved_dataset)


def testing(base_url, data, project_name, sample_name):
    data_train = {'brain': 'dataset3d/Task01_BrainTumour/Data_Train_Test', 'heart': 'dataset3d/Task02_Heart/Data_Train_Test', 'liver': 'dataset3d/Task03_Liver/Data_Train_Test'}
    model_dir = 'results' + data
    print(model_dir)

    in_dir = data_train[data.lower()]

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
        path = 'media/' + data + '/' + project_name + '/' + sample_name + '/output/'
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as ex:
            print("Exception: ", ex)
            
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
            plt.imsave(path + 'output_'+str(i)+'_1.png', test_patient["vol"][0, 0, :, :, i], cmap="gray")
            plt.imsave(path + 'output_'+str(i)+'_2.png', test_patient["seg"][0, 0, :, :, i] != 0)
            # plt.imsave('media/' + data + '/output/output_'+str(i)+'_3.png', test_outputs.detach().cpu()[0, 1, :, :, i])
            in1 = cv2.imread(path + 'output_'+str(i)+'_1.png')
            in2 = cv2.imread(path + 'output_'+str(i)+'_2.png')

            # Resize the images to have the same dimensions
            image1 = cv2.resize(in1, (in2.shape[1], in2.shape[0]))

            # Define the blending weight
            alpha = 0.5  # Adjust this value to control the visibility of each image

            # Perform alpha blending
            output_combined = cv2.addWeighted(image1, alpha, in2, 1 - alpha, 0)
            output_path = path + 'output_'+str(i)+'_3.png'  # Path to save the modified image
            output_combined = cv2.resize(output_combined, (500, 500))
            cv2.imwrite(output_path, output_combined)
        dicom_conversion(path, output_path, project_name, sample_name, data)
        print("Saved")
        file_urls = [base_url + filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename)) and str(filename.split(".")[0]).endswith('3')]
        sorted_list = sorted(file_urls, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return sorted_list

# data = "brain"#input("Enter the desired testing: ")
# project_name = "pro"#input("Enter the desired testing project_name: ")
# sample_name = "sam"#input("Enter the desired testing sample_name: ")

# testing(data, project_name, sample_name)