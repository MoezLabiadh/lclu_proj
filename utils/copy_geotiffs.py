import os
import shutil

def move_tif_files_to_root(root_directory):
    """
    Loops through subfolders of a directory and copy tif files.
    
    """
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".tif"):
                source_path = os.path.join(subdir, file)
                destination_path = os.path.join(root_directory, file)
                
                # Check if the file already exists in the root directory
                if not os.path.exists(destination_path):
                    shutil.copy(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")
                else:
                    print(f"File already exists in root: {destination_path}")



if __name__ == '__main__':
    wks= r'Q:\dss_workarea\mlabiadh\workspace\20241118_land_classification'
    
    #esa worldcover
    rootdir = os.path.join(wks, 'data', 'existing_data', 'esa', 'WORLDCOVER')
    
    move_tif_files_to_root(rootdir)
