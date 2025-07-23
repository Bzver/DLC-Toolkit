import os
import pandas as pd

def load_deeplabcut_csv(csv_file):
    """
    Loads a DeepLabCut CSV file and extracts labeled data.

    Args:
        csv_file (str): The path to the DeepLabCut CSV file.

    Returns:
        list: A list of labeled data from the CSV, or an empty list if an error occurs.
    """
    try:
        df = pd.read_csv(csv_file)
        # Assuming the relevant data starts from the 4th row (index 3) and 3rd column (index 2)
        data_labelled = df.iloc[3:, 2]
        data_list = data_labelled.tolist()
        return data_list
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_file} is empty.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the CSV: {e}")
        return []

def remove_extra_image(project_dir, data_list):
    """
    Removes image files from a project directory that are not present in the provided data list.

    Args:
        project_dir (str): The path to the project directory containing images.
        data_list (list): A list of image filenames that should be kept.
    """
    if not os.path.isdir(project_dir):
        print(f"Error: Project directory not found at {project_dir}")
        return

    if not data_list:
        print("Warning: Data list is empty. No images to compare against for deletion.")
        return

    images = [f for f in os.listdir(project_dir) if f.endswith(".png") and f.startswith("img")]
    deleted_images = []
    for image in images:
        if image not in data_list:
            image_full_path = os.path.join(project_dir, image)
            try:
                os.remove(image_full_path)
                deleted_images.append(image)
            except OSError as e:
                print(f"Error deleting image {image_full_path}: {e}")

    if deleted_images:
        print(f"Deleted {len(deleted_images)} images:\n {deleted_images}")
    else:
        print("No extra images found to delete.")

if __name__ == "__main__":
    csv_file = "D:/Project/DLC-Models/NTD/labeled-data/20250629r-conv/CollectedData_bezver.csv"
    project_dir = "D:/Project/DLC-Models/NTD/labeled-data/20250629r-conv/"
    data_list = load_deeplabcut_csv(csv_file)
    remove_extra_image(project_dir, data_list)