import os
import gc
from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model

def main():
    data_path = 'BirdSongsEurope/mp3/'
    df = load_data(data_path)

    df = preprocess_data(df)

    trainer, test_dataset = train_model(df)

    labels_list = sorted(list(df['label'].unique()))

    evaluate_model(trainer, test_dataset, labels_list)

    gc.collect()

if __name__ == "__main__":
    main()
