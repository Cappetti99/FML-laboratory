import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from tqdm import tqdm
import numpy as np
import time



def to_frequency_domain(x):
    return torch.fft.fft2(x)

def to_time_domain(x):
    return torch.fft.ifft2(x).real

class FrequencyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FrequencyConv, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.weights = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size, dtype=torch.cfloat))

    def forward(self, x):
        batch_size, _, H, W = x.shape
        kernel_padded = torch.zeros((batch_size, self.out_channels, self.in_channels, H, W), dtype=torch.cfloat, device=x.device)
        kernel_padded[:, :, :, :self.kernel_size, :self.kernel_size] = self.weights 
        kernel_freq = to_frequency_domain(kernel_padded)
        
        out_freq = x.unsqueeze(1) * kernel_freq
        return out_freq.sum(dim=2)
    
class NostroMinst(nn.Module):
    def __init__(self,conv1_channels, conv2_channels, fc1_size, kernel_size):
        super(NostroMinst, self).__init__()
        print("Modello= Nostro Minst")
        print(f"Kernel: {kernel_size}, Conv1: {conv1_channels}, Conv2: {conv2_channels}, FC1: {fc1_size}")

        self.conv1 = FrequencyConv(1, conv1_channels, kernel_size)
        self.conv2 = FrequencyConv(conv1_channels, conv2_channels, kernel_size)
        self.fc1 = None
        self.fc2 = nn.Linear(fc1_size, 10)
        self.flatten_dim=None
        self.fc1_size = fc1_size

        
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.abs(x)
        x = self.conv2(x)
        x = torch.abs(x)
        if self.flatten_dim is None:
            print("Dimensione dopo convoluzione:", x.shape)
            self.flatten_dim = x.shape[1] * x.shape[2] * x.shape[3]  # Calcoliamo la dimensione corretta
            self.fc1 = nn.Linear(self.flatten_dim, self.fc1_size).to(x.device)  # Inizializziamo `fc1` dinamicamente

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_model(conv1_channels=8, conv2_channels=16, fc1_size=128, kernel_size=3,number_epochs=5):

    start_time = time.time()  # Salva il tempo di inizio

    saved_data = torch.load('./dataset/transformed_mnist.pt',weights_only=False)

    train_images = saved_data['train_images']
    train_labels = saved_data['train_labels']
    test_images = saved_data['test_images']
    test_labels = saved_data['test_labels']

    trainset = torch.utils.data.TensorDataset(train_images, train_labels)
    testset = torch.utils.data.TensorDataset(test_images, test_labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    print("Dataset loaded successfully.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = NostroMinst(conv1_channels, conv2_channels, fc1_size, kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in tqdm(range(number_epochs)):
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoca [{epoch+1}/{number_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Accuracy sui test: {test_accuracy:.2f}%')
    end_time = time.time()  # Salva il tempo di fine
    execution_time = end_time - start_time  # Calcola il tempo trascorso

    print(f"Tempo di esecuzione: {execution_time:.4f} secondi")

if __name__ == "__main__":
    print("➡️ Avvio training con parametri di default")
    train_model()