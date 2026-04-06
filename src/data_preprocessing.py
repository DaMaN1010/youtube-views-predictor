import pandas as pd

def load_data(path):
    """
    Load dataset from a CSV file.

    Parameters:
    path (str): Path to the dataset

    Returns:
    DataFrame: Loaded dataset
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    Clean and prepare the dataset.

    Steps:
    - Convert publish_time to datetime
    - Remove missing values
    """

    # Convert publish_time column to datetime format
    df['publish_time'] = pd.to_datetime(df['publish_time'])

    # Remove rows with missing values
    df = df.dropna()

    return df
