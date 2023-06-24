
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

import shutil
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
import boto3
from monai.inferers import sliding_window_inference
import random
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
    
    s3_client = boto3.client('s3', aws_access_key_id='AKIA3IFSKFQQ2E3G76UO', aws_secret_access_key='K+KNXT03O90pzHL7Hn93a1XWd+uPqXqH8Y7K76+Q')
    
    bucket_name = 'hiringuard'
    unique_folder_name = 'googgerit/'  + project_name + '/' + data + '/' + sample_name

    file_name = 'output.dcm'
    local_file_path = output_path

    # Step 6: Upload the file to S3 with the unique folder
    s3_key = os.path.join(unique_folder_name, file_name)
    s3_client.upload_file(local_file_path, bucket_name, s3_key)
    s3_client.put_object_acl(
        ACL='public-read',
        Bucket=bucket_name,
        Key=s3_key
    )
    # s3_resource.meta.client.upload_file(local_file_path, bucket_name, s3_key)

def get_dicom_file_path(project_name, data, sample_name):
    s3_client = boto3.client('s3', aws_access_key_id='AKIA3IFSKFQQ2E3G76UO', aws_secret_access_key='K+KNXT03O90pzHL7Hn93a1XWd+uPqXqH8Y7K76+Q')

    bucket_name = 'hiringuard'
    unique_folder_name = 'googgerit/'  + project_name + '/' + data + '/' + sample_name
    file_name = 'output.dcm'

    file_name = 'output.dcm'

    # Step 6: Upload the file to S3 with the unique folder
    s3_key = os.path.join(unique_folder_name, file_name)

    # Step 3: Generate the presigned URL for the file
    expiration_time = 3600  # Time in seconds for the URL to be valid

    # Step 5: Generate the URL for the uploaded file
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': s3_key},
        ExpiresIn=3600  # Time in seconds for the URL to be valid
    )

    # Step 6: Print or use the presigned URL
    print("Presigned URL:", presigned_url)
    return presigned_url

def testing(base_url, vol_path, seg_path, input_file, data, project_name, sample_name):
    data_train = {'brain': 'dataset3d/Task01_BrainTumour/Data_Train_Test', 'heart': 'dataset3d/Task02_Heart/Data_Train_Test', 'liver': 'dataset3d/Task03_Liver/Data_Train_Test'}
    model_dir = 'results' + data
    print(model_dir)

    in_dir = data_train[data.lower()]
    folder_path = in_dir + '/TrainSegmentation'  
    destination_path = ""
    directory, filename = os.path.split(input_file)
    print("filename", filename)
    number = str(filename.split(".")[0]).split("_")[1]  # Assuming the format is "BRATS_001_0000.nii"
    print("number", number)
    file_test_seg = []
    # Check if the number exists in the other folder
    
    # path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    # path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = [input_file]
    # train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name} for image_name in path_test_volumes]
    test_files = test_files[0:9]

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            AddChanneld(keys=["vol"]),
            Spacingd(keys=["vol"], pixdim=(1.5,1.5,1.0), mode=("bilinear")),
            Orientationd(keys=["vol"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol'], source_key='vol'),
            Resized(keys=["vol"], spatial_size=[128,128,64]),   
            ToTensord(keys=["vol"]),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    device = torch.device("cpu")
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
        os.path.join(model_dir, "best_metric_model.pth"), map_location=torch.device('cpu')))
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
        path = '/home/ubuntu/mystorage/media/' + project_name + '/' + data + '/' + sample_name + '/output/'
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as ex:
            print("Exception: ", ex)
            
        for i in range(61):
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_patient["vol"][0, 0, :, :, i] != 0)
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
            plt.imsave(path + 'output_'+str(i)+'_1.png', test_patient["vol"][0, 0, :, :, i], cmap="gray")
            plt.imsave(path + 'output_'+str(i)+'_2.png', test_outputs.detach().cpu()[0, 1, :, :, i])
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
            # plt.imsave(path + 'output_'+str(i)+'_4.png', output_combined)
            # plt.savefig(path + 'output_'+str(i)+'_5.png')
        image1 = cv2.imread(path + 'output_'+str(i)+'_1.png')
        image2 = cv2.imread(path + 'output_'+str(i)+'_2.png')
        image3 = cv2.imread(path + 'output_'+str(i)+'_3.png')
        image1 = cv2.resize(image1, (500, 500))
        image2 = cv2.resize(image2, (500, 500))
        image3 = cv2.resize(image3, (500, 500))
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2)
        # Plot the images in the subplots
        axes[0].imshow(image1, cmap='gray')
        axes[0].set_title('Image')
        # axes[1].imshow(image2, cmap='gray')
        # axes[1].set_title('Label')
        axes[1].imshow(image3, cmap='gray')
        axes[1].set_title('Output')
        # Adjust the layout
        plt.tight_layout()
        # Show the plot
        output_path_res = path.replace('/output/', '/') + "final_output.png"
        plt.savefig(output_path_res)
        dicom_conversion(path, output_path, project_name, sample_name, data)
        print("Saved")
        file_urls = [base_url + filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename)) and str(filename.split(".")[0]).endswith('3')]
        sorted_list = sorted(file_urls, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return sorted_list


def get_output_url(base_url, data, project_name, sample_name):
    path = 'media/' + project_name + '/' + data + '/' + sample_name + '/output/'
    try:
        print(os.listdir('/home/ubuntu/mystorage/' + path))
        file_urls = [base_url + '/static/' + path + filename for filename in os.listdir('/home/ubuntu/mystorage/'+path) if (os.path.isfile(os.path.join('/home/ubuntu/mystorage/'+path, filename)) and str(filename.split(".")[0]).endswith('3'))]
        sorted_list = sorted(file_urls, key=lambda x: int(x.split('_')[1].split('.')[0]))
        # print(sorted_list)
        # print("*******************************")
        return sorted_list
    except Exception as ex:
        print("Exception: ", ex)
        return []
# data = "brain"#input("Enter the desired testing: ")
# project_name = "pro"#input("Enter the desired testing project_name: ")
# sample_name = "sam"#input("Enter the desired testing sample_name: ")

# testing(data, project_name, sample_name)