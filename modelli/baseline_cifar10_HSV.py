import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ClassicCNN(nn.Module):
    def __init__(self, conv1_channels, conv2_channels, fc1_size, kernel_size):
        super(ClassicCNN, self).__init__()
        print(f"Kernel: {kernel_size}, Conv1: {conv1_channels}, Conv2: {conv2_channels}, FC1: {fc1_size}")
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size=kernel_size, padding=2)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=2)

        # Determina la dimensione dell'input per il layer fully connected
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
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

def train_model(conv1_channels=16, conv2_channels=32, fc1_size=128, kernel_size=5, number_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ClassicCNN(conv1_channels, conv2_channels, fc1_size, kernel_size).to(device)

    saved_data = torch.load('./dataset/cifar10_HSV.pt', weights_only=False)

    train_images = saved_data['train_images']
    train_labels = saved_data['train_labels']
    test_images = saved_data['test_images']
    test_labels = saved_data['test_labels']

    trainset = torch.utils.data.TensorDataset(train_images, train_labels)
    testset = torch.utils.data.TensorDataset(test_images, test_labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    print("Dataset loaded successfully.")
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

if __name__ == "__main__":
    print("➡️ Eseguo il training con i parametri di default")
    train_model()