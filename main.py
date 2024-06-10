import os
import gc
from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model

def main():
    print("Loading data...")
    data_path = 'BirdSongsEurope/mp3/'
    df = load_data(data_path)
    print("Data loading complete.")
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    print("Data preprocessing complete.")
    
    print("Training model...")
    trainer, test_dataset = train_model(df)
    print("Model training complete.")
    
    labels_list = sorted(list(df['label'].unique()))
    
    print("Evaluating model...")
    evaluate_model(trainer, test_dataset, labels_list)
    print("Model evaluation complete.")
    
    gc.collect()
    print("Script execution complete.")

if __name__ == "__main__":
    main()
