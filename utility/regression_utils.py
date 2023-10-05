import os
import sys
import zipfile
import random
from io import StringIO
from torchinfo import summary


def ensure_zip_file(file_path):
    """Check if the specified file is a zip file."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            pass  # File is a valid zip file, do nothing
    except zipfile.BadZipFile:
        print(f'Error: The specified file "{file_path}" is not a valid zip file.')
        sys.exit(1)


def create_directories(path):
    directory = os.path.normpath(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_dataset_sizes_enough(training_tagged, training_other, validation_tagged, validation_other):
    if (len(training_tagged.dataset) > 0 and
       len(training_other.dataset) > 0 and
       len(validation_tagged.dataset) > 0 and
       len(validation_other.dataset) > 0):
        return True
    return False


def balance_other_dataset(tagged_dataset, other_dataset, is_balance=False):
    if is_balance and len(other_dataset.dataset) > len(tagged_dataset.dataset):
        return random.sample(other_dataset.dataset, len(tagged_dataset.dataset))
    return other_dataset.dataset


def delete_all_files_in_folder(folder_path):
    # Iterate over all the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the current path is a file
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)


def torchinfo_summary(model, data):
    # Capturing torchinfo output
    string_buffer = StringIO()
    sys.stdout = string_buffer

    # Running torchinfo
    summary(model, input_size=data[0].size(), col_names=("input_size", "output_size", "num_params"))

    # Get the captured output as a string
    summary_string = string_buffer.getvalue()

    # Restore the standard output
    sys.stdout = sys.__stdout__

    return summary_string
