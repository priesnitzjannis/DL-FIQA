import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import ViTModel, ViTFeatureExtractor
import argparse

# Define the FingerprintDataset class
class FingerprintDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image = Image.open(image_path)
        image = image.convert('RGB')  # Convert to RGB color
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Define the ViT model
class ViTModelForRegression(torch.nn.Module):
    def __init__(self):
        super(ViTModelForRegression, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = torch.nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, x):
        outputs = self.vit(x)
        pooled_output = outputs.pooler_output
        outputs = self.fc(pooled_output)
        return outputs

def train(model_config, device):
    # Load the dataset
    csv_file = model_config['csv_file']
    dataset = pd.read_csv(csv_file)

    # Split the dataset into training and evaluation sets
    train_data, eval_data = train_test_split(dataset, test_size=model_config['test_size'], random_state=42)

    # Create FingerprintDataset instances for training and evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = FingerprintDataset(train_data, transform=transform)
    eval_dataset = FingerprintDataset(eval_data, transform=transform)

    # Create data loaders for training and evaluation
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the ViT model
    model = ViTModelForRegression()
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    # Train the model
    for epoch in range(model_config['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(train_dataloader)}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            total_mse = 0
            for batch in tqdm(eval_dataloader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)
                outputs = model(inputs)
                mse = criterion(outputs, labels)
                total_mse += mse.item()
            print(f'Epoch {epoch+1}, Evaluation MSE: {total_mse / len(eval_dataloader)}')

    torch.save(model.state_dict(), model_config['model_name'] + ".pth")

def test(model_config, device):
    # Load the model
    model = ViTModelForRegression()
    model.load_state_dict(torch.load(model_config['model_name']))
    model.to(device)

    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Use the model for prediction
    def predict_quality(image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')  # Convert to RGB color
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            quality = output.item()
            return quality

    # Example usage
    csv_file = model_config['csv_file']
    df = pd.read_csv(csv_file)
    df.columns = ["Filename"]
    new_data = []

    for file_path in tqdm(df['Filename']):
        quality = predict_quality(file_path)
        #print(f'Predicted quality: {quality}')
        new_data.append([file_path, quality])
        # Convert the list of images to a numpy array

    new_data = np.array(new_data)
    df = pd.DataFrame(new_data)
    df.to_csv(f"./{model_config['model_name'][:-4]}_predictions.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description='Train or test a ViT model for regression')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode to run the script in')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing the dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to use for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model for')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_config = {
        'model_name': args.model_name,
        'csv_file': args.csv_file,
        'test_size': args.test_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate
    }

    if args.mode == 'train':
        train(model_config, device)
    elif args.mode == 'test':
        test(model_config, device)

if __name__ == '__main__':
    main()
