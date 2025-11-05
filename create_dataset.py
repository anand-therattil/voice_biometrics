import pandas as pd 
import numpy as np
import os 
import sys 
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_csv(df, path):
    try:
        df.to_csv(path, index=False)
        print(f"CSV file created at: {path}")
    except Exception as e:
        print(f"Error creating CSV file: {e}")


def load_csv(path):
    try:
        df = pd.read_csv(path)
        df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')

        print(f"CSV file loaded from: {path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def main():
    data_path = os.path.join(BASE_DIR, "datasets/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt")
    print(f"{data_path=}")

    df = load_csv(data_path)
    if df is not None:

        df.drop(df.columns[[2,3,4,6]], axis=1, inplace=True)
        df.drop(df[df[7]!="eval"].index, inplace=True)
        df.columns = ['speaker_id', 'audio_file', 'label', 'subset']
        print(df.head())
        df['file_path'] = df['audio_file'].apply(lambda x: os.path.join(BASE_DIR, "datasets/ASVspoof2021_LA_eval/flac/", x + ".flac"))

        output_path = os.path.join(BASE_DIR, "data.csv")
        create_csv(df, output_path)

if __name__ == "__main__":
    main()


