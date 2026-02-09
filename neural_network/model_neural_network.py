import joblib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from visualization_utils import evaluate_and_plot
import numpy as np

# 1. Define Model Architecture
class BinaryIntrusionNet(nn.Module):
    def __init__(self, input_dim):
        super(BinaryIntrusionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),       
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 2) # Output: [Benign_Score, Attack_Score]
        )

    def forward(self, x):
        return self.network(x)

# 2. Wrapper for compatibility with our plotting function
class PyTorchWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.last_probs = None # Store probabilities for ROC curve
    
    def predict(self, X):
        # We calculate probs here to save them for ROC
        self.model.eval()
        # Process in chunks to avoid GPU OOM on testing
        predictions = []
        probabilities = []
        
        chunk_size = 10000
        with torch.no_grad():
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i+chunk_size]
                chunk_t = torch.FloatTensor(chunk).to(self.device)
                logits = self.model(chunk_t)
                probs = torch.softmax(logits, dim=1)
                
                probabilities.extend(probs[:, 1].cpu().numpy()) # Probability of Attack
                predictions.extend(torch.argmax(probs, dim=1).cpu().numpy())
                
        self.last_probs = np.array(probabilities)
        return np.array(predictions)

def run():
    print("Loading data...")
    X_train = joblib.load('processed_data/X_train.pkl')
    X_test = joblib.load('processed_data/X_test.pkl')
    y_train = joblib.load('processed_data/y_train.pkl')
    y_test = joblib.load('processed_data/y_test.pkl')
    label_mapping = joblib.load('processed_data/label_mapping.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Calculate Class Weights for Imbalance
    # Weight = Total_Samples / (Num_Classes * Count_Class)
    # Roughly: Benign=0.5, Attack=6.5. We normalize relative to Benign.
    neg_counts = (y_train == 0).sum()
    pos_counts = (y_train == 1).sum()
    pos_weight = neg_counts / pos_counts
    print(f"Imbalance Ratio: 1 Attack for every {pos_weight:.2f} Benign flows.")
    
    # We use this weight in the Loss Function
    class_weights = torch.tensor([1.0, float(pos_weight)]).to(device)

    # Convert Train Data to Tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train.values).to(device)

    model = BinaryIntrusionNet(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # <--- HANDLE IMBALANCE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Batch Size: 4096 is good for 7 million rows on GPU
    train_data = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(train_data, batch_size=4096, shuffle=True)

    print("Training Neural Network...")
    start_time = time.time()
    epochs = 5 # 5 is enough for this much data
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")

    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    # Wrap and Evaluate
    wrapped_model = PyTorchWrapper(model, device)
    evaluate_and_plot(wrapped_model, X_test, y_test, "Neural Network (MLP)", label_mapping)

if __name__ == "__main__":
    run()