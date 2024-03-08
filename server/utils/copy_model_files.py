import os
from rosepetal.file_management.file_manager import FileManager

def copy_model_files(model_path, model_files):
    for key, value in model_files.items():
        FileManager.write(data=value, file_path=FileManager.merge_paths(model_path, key))