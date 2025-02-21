import baseline_mnist 
import nostro_cifar10_HVS

# Parametri personalizzati
conv1_out_channels = 8  
conv2_out_channels = 16  
fc1_neurons = 128  
kernel_size = 3  
epochs = 1

# Esegui il training con i parametri personalizzati
if __name__ == "__main__":
    print("➡️ Avvio training con parametri personalizzati")
    #baseline_mnist.train_model(conv1_out_channels, conv2_out_channels, fc1_neurons, kernel_size, epochs)
    #baseline_cifar10_RGB.train_model(conv1_out_channels, conv2_out_channels, fc1_neurons, kernel_size, epochs)
    nostro_cifar10_HVS.train_model(conv1_out_channels, conv2_out_channels, fc1_neurons, kernel_size, epochs)