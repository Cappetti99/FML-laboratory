import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time


def create_model(conv1_channels, conv2_channels, fc1_size, kernel_size):
    class ClassicCNN(nn.Module):
        def __init__(self):
            super(ClassicCNN, self).__init__()
            print("Modello= Baseline Minst")
            print(f"Kernel: {kernel_size}, Conv1: {conv1_channels}, Conv2: {conv2_channels}, FC1: {fc1_size}")
            self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size, padding=2)
            self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size, padding=2)

            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, 28, 28)
                dummy_output = self.conv2(self.conv1(dummy_input))
                self.flatten_dim = dummy_output.numel()

            self.fc1 = nn.Linear(self.flatten_dim, fc1_size)
            self.fc2 = nn.Linear(fc1_size, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    return ClassicCNN()

# Funzione per addestrare il modello
def train_model(conv1_channels=16, conv2_channels=32, fc1_size=256, kernel_size=3, number_epochs=5):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(conv1_channels, conv2_channels, fc1_size, kernel_size).to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

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

    # Valutazione sul test set
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

# Permette di eseguire il codice principale solo se avviato direttamente
if __name__ == "__main__":
    print("➡️ Avvio training con parametri di default")
    train_model()