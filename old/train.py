from model import SpeechDisfluencyClassifier
from data_preprocessor import DataPreprocessor
from model_utils import save_model, calculate_accuracy
import torch.optim as optim
import torch.nn as nn
import torch

def train_model(train_loader, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")

def main():
    # Initialize data preprocessor, model, etc.
    # ...

    # Train the model
    model = SpeechDisfluencyClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    train_model(train_loader, model, criterion, optimizer, epochs)

    # Save the trained model
    save_model(model, 'trained_speech_disfluency_model.pth')

if __name__ == "__main__":
    main()
