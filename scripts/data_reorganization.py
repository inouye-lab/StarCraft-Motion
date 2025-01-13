import os
import shutil

def reorganize_directory(source_directory, target_directory):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_directory):
        source_file_path = os.path.join(source_directory, filename)

        # Check if the current item is a file
        if os.path.isfile(source_file_path):
            # Extract the base name (filename without extension or additional tags)
            base_name = filename.split('.')[0].split('_')[0]

            # Create a new folder inside the target directory with the base name if it doesn't already exist
            folder_path = os.path.join(target_directory, base_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Determine the new name based on the file type and existing name
            if filename.endswith('.npz'):
                new_filename = 'track.npz'
            elif filename.endswith('.mp4') and '_vis_1' in filename:
                new_filename = 'fog_1.mp4'
            elif filename.endswith('.mp4') and '_vis_2' in filename:
                new_filename = 'fog_2.mp4'
            elif filename.endswith('.mp4') and '_vis' not in filename:
                new_filename = 'video.mp4'
            else:
                continue  # Skip any file that doesn't fit the criteria

            # Copy and rename the file to the appropriate folder in the target directory
            target_file_path = os.path.join(folder_path, new_filename)
            shutil.copy2(source_file_path, target_file_path)

# Example usage
source_directory = "/local/scratch/a/bai116/datasets/StarCraftMotion"
target_directory = '/local/scratch/a/bai116/datasets/StarCraftMotion_v0.2'  # Replace with the path to your target directory
reorganize_directory(source_directory, target_directory)
