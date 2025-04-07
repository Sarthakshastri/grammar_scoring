import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print("Extract_mfcc\n")
def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten()

print("Processing dataset\n")
def process_dataset(audio_dir, csv_file):
    df = pd.read_csv(csv_file)
    features = []
    labels = []
    for _, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        mfcc = extract_mfcc(file_path)
        features.append(mfcc)
        labels.append(row['label'])  # Or 'grammar_score' if that's the column name
    return np.array(features), np.array(labels)

print("Done with the preprocessing\n")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

print("Train Model\n")
def train_model(audio_dir, csv_file, epochs=50, batch_size=16):
    X, y = process_dataset(audio_dir, csv_file)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = MLPRegressor(input_dim=X.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

    return model, X.shape[1]

print("Test the Model\n")
def predict_from_trained_model(audio_dir_unlabeled, audio_dir_train, csv_file_train, output_csv="results/predicted_scores.csv"):
    model, input_dim = train_model(audio_dir_train, csv_file_train)
    audio_files = [f for f in os.listdir(audio_dir_unlabeled) if f.endswith(".wav")]
    print(f"\nFound {len(audio_files)} test audio files.")
    features = []
    file_names = []
    for file in audio_files:
        path = os.path.join(audio_dir_unlabeled, file)
        mfcc = extract_mfcc(path)
        features.append(mfcc)
        file_names.append(file)
    X = torch.FloatTensor(np.array(features))
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    df = pd.DataFrame({
        'filename': file_names,
        'predicted_score': np.round(preds, 1)
    })
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    return df

if __name__ == "__main__":
    TRAIN_AUDIO_DIR = "C:/Users/hp/Desktop/M.Tech/Project/SHL Project/audios_train"
    TRAIN_CSV = "C:/Users/hp/Desktop/M.Tech/Project/SHL Project/train.csv"
    TEST_AUDIO_DIR = "C:/Users/hp/Desktop/M.Tech/Project/SHL Project/audios_test"
    OUTPUT_CSV="C:/Users/hp/Desktop/M.Tech/Project/SHL Project/final_grammar_score.csv"

    predict_from_trained_model(
        audio_dir_unlabeled=TEST_AUDIO_DIR,
        audio_dir_train=TRAIN_AUDIO_DIR,
        csv_file_train=TRAIN_CSV,
        output_csv=OUTPUT_CSV
    )
