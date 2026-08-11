import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

class MLPTrainer:
    def __init__(self, input_size=3, hidden_size=128, output_size=2, device="cpu", gpu_id=0):
        if torch.cuda.is_available() and device == 'cuda':
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device('cpu')
        self.model = MLPModel(input_size, hidden_size, output_size).to(self.device)

    def train(self, X, y, epochs=10, decay_rate=0.9):
        """
        X: features, y: targets
        decay_rate: standard value (0.9). Lower means forgetting old data faster.
        """
        print(f'Using MLP ReLU with Exponential Decay.')
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
        
        # Log-transform to stabilize training
        y_np = np.log1p(np.abs(y_np)) * np.sign(y_np)

        # Generate sample weights (Exponential decay)
        # Newest data gets weight 1.0, older data gets decay_rate^n
        num_samples = len(X_np)
        weights = np.array([decay_rate**(num_samples - i - 1) for i in range(num_samples)], dtype=np.float32)
        
        # Convert to Tensors
        X_tensor = torch.tensor(X_np).to(self.device)
        y_tensor = torch.tensor(y_np).to(self.device)
        w_tensor = torch.tensor(weights).to(self.device)

        # Create DataLoader (shuffle=False to maintain chronological order)
        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # 'none' reduction allows applying manual weights per sample
        criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y, batch_w in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                # Calculate raw MSE per sample
                loss = criterion(outputs, batch_y)
                
                # Apply weights: (Loss * Weight).mean()
                # batch_w.unsqueeze(1) aligns weight shape with loss shape
                weighted_loss = (loss * batch_w.unsqueeze(1)).mean()
                
                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item()
            
            print(f"[MLP] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

    def predict(self, X_input):
        self.model.eval()
        X_np = np.array(X_input, dtype=np.float32)
        if len(X_np.shape) == 1:
            X_np = X_np.reshape(1, -1)
            
        input_tensor = torch.tensor(X_np).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            return output.cpu().numpy()