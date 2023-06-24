import pydicom
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from PIL import Image

# Load the image using Pillow
image_path = "output_blended.png"  # Path to your image file
image = Image.open(image_path)

# Convert the image to grayscale if it has more than one channel
image = image.convert("L")

# Create a new DICOM dataset
dataset = Dataset()

# Set the necessary DICOM attributes
dataset.PatientName = "Vishal S"
dataset.StudyDescription = "Brain Segmentation"
dataset.SeriesDescription = "Converted"
dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
dataset.SOPInstanceUID = generate_uid()
dataset.Modality = 'CT'
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
output_path = "p10output.dcm"  # Path to save the converted DICOM file
dataset.save_as(output_path)

# Read the saved DICOM file and print its information
saved_dataset = dcmread(output_path, force=True)
print(saved_dataset)
