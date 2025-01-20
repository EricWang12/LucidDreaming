import os
import shutil

def copy_save_folders_with_subdirs(src, dest):
    """
    Copies all 'save' subfolders along with their subdirectories and files 
    from 'src' directory to 'dest' directory.
    """
    for root, dirs, files in os.walk(src):
        # Check if current folder is a 'save' folder
        if 'save' in root.split(os.sep):
            # Determine the path to create in the destination
            dest_path = os.path.join(dest, os.path.relpath(root, src))
            if os.path.exists(dest_path):
                continue
            # Create the destination path and copy all contents
            os.makedirs(dest_path, exist_ok=True)
            for file in files:
                # breakpoint()
                shutil.copy2(os.path.join(root, file), dest_path)

# # Resetting the destination folder
# if os.path.exists(destination_folder):
#     shutil.rmtree(destination_folder)
# os.makedirs(destination_folder)

# # Copying the 'save' folders along with their subdirectories and files
# copy_save_folders_with_subdirs(source_folder, destination_folder)

# # Return a success message
# "Save folders along with their subdirectories and files have been successfully copied."

# Source and destination paths
source_folder = 'outputs/multi_gen/magic3d/extra'
destination_folder = './new_output/multi_gen/magic3d/extra'

# Creating the destination folder if it does not exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Copying the 'save' folders
copy_save_folders_with_subdirs(source_folder, destination_folder)

# Return a success message
print("Save folders have been successfully copied.")
def zip_directory(folder_path, zip_name):
    """
    Zips the specified folder into an archive with the given name.
    """
    shutil.make_archive(zip_name, 'zip', folder_path)

# Path for the zip file (without the .zip extension)
zip_file_path = './new_output_CHEN'

# Creating the zip file
# zip_directory('./new_output', zip_file_path)