import os
import zipfile

# Path to the folder containing U-Net predictions
predictions_folder = 'dataset/test1/unet_balanced'
zip_path = 'dataset/test1/predictions.zip'

# Create a zip file and add all files from the predictions folder
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(predictions_folder):
        for file in files:
            file_path = os.path.join(root, file)
            # Add file to zip, store only the filename (not full path)
            zipf.write(file_path, arcname=file)

print(f"All prediction images zipped to {zip_path}")