import kagglehub
import shutil
import os

def download_data():
    """
    Download dataset from Kaggle using kagglehub.

    This avoids uploading large files to GitHub and keeps the project clean.
    """

    # Download dataset
    path = kagglehub.dataset_download("abdeltawabali/usvideos-csv")

    print("Dataset downloaded at:", path)

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Move CSV files into our data/ folder
    for file in os.listdir(path):
        if file.endswith(".csv"):
            source = os.path.join(path, file)
            destination = os.path.join("data", file)

            shutil.copy(source, destination)
            print(f"Copied {file} to data/ folder")

    print("✅ Dataset is ready inside the data/ folder.")


if __name__ == "__main__":
    download_data()
