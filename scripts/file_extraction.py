import os
import shutil

def extract_specific_files_from_folders(source_folder, destination_folder, file_names, folder_names=None):
    total_files = 0
    copied_files = 0

    def copy_progress(src, dst):
        nonlocal copied_files
        shutil.copy2(src, dst)
        copied_files += 1
        percentage = round((copied_files / total_files) * 100, 2)
        print(f"Copying: {percentage}%, [{copied_files}/{total_files}]")

    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):
            destination_subfolder = os.path.join(destination_folder, folder_name)
            os.makedirs(destination_subfolder, exist_ok=True)

            for root, dirs, files in os.walk(folder_path):
                total_files += len(files)
                for file in files:
                    if file in file_names:
                        source_path = os.path.join(root, file)
                        destination_path = os.path.join(destination_subfolder, file)
                        copy_progress(source_path, destination_path)

            if folder_names is not None:
                for dir_name in folder_names:
                    source_dir = os.path.join(folder_path, dir_name)
                    destination_dir = os.path.join(destination_subfolder, dir_name)
                    shutil.copytree(source_dir, destination_dir, copy_function=copy_progress)

source_folder = 'exp/traffic_thermal_yellow_64'
destination_folder = '/media/hkuit164/Backup/xjl/train_data/traffic_thermal_yellow_64'
file_names = ['model_cfg.json', 'data_cfg.json', 'option.pkl']
folder_names = []

extract_specific_files_from_folders(source_folder, destination_folder, file_names, folder_names)
