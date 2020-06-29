import gzip
import json

import pandas as pd

from src.constants import DATA_FOLDER


def get_amazon_dataset():
    if not (DATA_FOLDER / 'reviews.csv').exists():
        df = pd.DataFrame.from_records(map(json.loads, gzip.open(DATA_FOLDER / 'reviews_Home_and_Kitchen_5.json.gz')))
        df[['reviewText', 'overall']].to_csv(DATA_FOLDER / 'reviews.csv')
    df = pd.read_csv(DATA_FOLDER / 'reviews.csv').set_index("Unnamed: 0")
    return df


def get_balanced_dataset():
    df = get_amazon_dataset()
    if not (DATA_FOLDER / 'reviews_small.csv').exists():
        df = df.dropna()
        new_df_5 = df[df['overall'] == 5][:20000]
        new_df_4 = df[df['overall'] == 4][:20000]
        new_df_3 = df[df['overall'] == 3][:20000]
        new_df_2 = df[df['overall'] == 2][:20000]
        new_df_1 = df[df['overall'] == 1][:20000]
        pdList = [new_df_1, new_df_2, new_df_3, new_df_4, new_df_5]
        df = pd.concat(pdList)
        df[['reviewText', 'overall']].to_csv(DATA_FOLDER / 'reviews_small.csv')
    df = pd.read_csv(DATA_FOLDER / 'reviews_small.csv').set_index("Unnamed: 0")
    return df