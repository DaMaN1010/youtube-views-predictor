import numpy as np

def create_features(df):
    """
    Create new features from raw data.

    Raw data is often not useful directly.
    We extract meaningful information for the model.
    """

    # Time-based features
    # Extract hour, day, and month from publish time
    df['publish_hour'] = df['publish_time'].dt.hour
    df['publish_day'] = df['publish_time'].dt.day
    df['publish_month'] = df['publish_time'].dt.month

    # Title-based features
    # Length of the title (number of characters)
    df['title_length'] = df['title'].apply(len)

    # Number of words in the title
    df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))

    #  Tags feature
    # Count how many tags a video has
    df['num_tags'] = df['tags'].apply(lambda x: len(str(x).split('|')))

    # Target transformation
    # Log transformation helps reduce skewness in views
    df['log_views'] = np.log1p(df['views'])

    return df
