import nibabel as nib
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

# Load the NIfTI file
nifti_path = "abc.nii.gz"  # Path to your NIfTI file
nifti_image = nib.load(nifti_path)
nifti_data = nifti_image.get_fdata()

# Create a new DICOM dataset
dataset = Dataset()
dataset.is_little_endian = True
dataset.is_implicit_VR = True

# Set the necessary DICOM attributes
dataset.PatientName = "Vishal S"
dataset.StudyDescription = "Brain Segmentation"
dataset.SeriesDescription = "Converted"
dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
dataset.SOPInstanceUID = generate_uid()
dataset.Modality = 'MR'
dataset.Rows, dataset.Columns, _ = nifti_data.shape
dataset.PixelRepresentation = 0  # Unsigned integer
dataset.BitsAllocated = 16
dataset.BitsStored = 16
dataset.HighBit = 15
dataset.PixelData = nifti_data.tobytes()

# Save the DICOM dataset to a file
output_path = "output.dcm"  # Path to save the converted DICOM file
pydicom.dcmwrite(output_path, dataset)
