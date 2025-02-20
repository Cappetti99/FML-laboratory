import baseline_mnist 
import baseline_cifar10

# Parametri personalizzati
conv1_out_channels = 16  
conv2_out_channels = 32  
fc1_neurons = 128  
kernel_size = 3  
epochs = 6

# Esegui il training con i parametri personalizzati
if __name__ == "__main__":
    print("➡️ Avvio training con parametri personalizzati")
    #baseline_mnist.train_model(conv1_out_channels, conv2_out_channels, fc1_neurons, kernel_size, epochs)
    baseline_cifar10.train_model(conv1_out_channels, conv2_out_channels, fc1_neurons, kernel_size, epochs)
