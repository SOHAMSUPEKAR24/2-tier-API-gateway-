import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EDLModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(EDLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        evidence = F.softplus(self.fc3(x)) # Evidence must be non-negative
        return evidence

def edl_loss(evidence, target, num_classes=2, epoch_num=1):
    """
    Sum of squared error loss for Evidential Deep Learning
    """
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    
    # One-hot encode target
    y = F.one_hot(target, num_classes=num_classes).float()
    
    # Loss components
    err = torch.sum((y - prob)**2, dim=1, keepdim=True)
    var = torch.sum(prob * (1 - prob) / (S + 1), dim=1, keepdim=True)
    
    # KL divergence regularization could be added here
    loss = err + var
    return torch.mean(loss)

class Tier1EDL:
    def __init__(self, input_dim=9):
        self.model = EDLModel(input_dim=input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
    def fit(self, X, y, epochs=10):
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            evidence = self.model(X_t)
            loss = edl_loss(evidence, y_t)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            evidence = self.model(X_t)
            alpha = evidence + 1
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            return torch.argmax(prob, dim=1).numpy()
            
    def get_uncertainty(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            evidence = self.model(X_t)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            uncertainty = 2 / S # num_classes / S
            return uncertainty.numpy().flatten()
