import baseline_mnist 
import nostro_minst
import baseline_cifar10_RGB
import baseline_cifar10_HSV
import nostro_cifar10_RGB
import nostro_cifar10_HVS
# Parametri personalizzati
conv1_out_channels1 = 8  
conv2_out_channels1 = 16
conv1_out_channels2 = 16  
conv2_out_channels2 = 32    
fc1_neurons = 256  
kernel_size1 = 3
kernel_size2 = 7    
epochs = 1

# Esegui il training con i parametri personalizzati
if __name__ == "__main__":
    print("➡️ Avvio training con parametri personalizzati")
    baseline_mnist.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size1, epochs)
    baseline_mnist.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size1, epochs)
    baseline_mnist.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size2, epochs)
    baseline_mnist.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size2, epochs)

    nostro_minst.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size1, epochs)
    nostro_minst.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size1, epochs)
    nostro_minst.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size2, epochs)
    nostro_minst.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size2, epochs)

    baseline_cifar10_RGB.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size1, epochs)
    baseline_cifar10_RGB.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size1, epochs)
    baseline_cifar10_RGB.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size2, epochs)
    baseline_cifar10_RGB.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size2, epochs)

    baseline_cifar10_HSV.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size1, epochs)
    baseline_cifar10_HSV.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size1, epochs)
    baseline_cifar10_HSV.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size2, epochs)
    baseline_cifar10_HSV.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size2, epochs)

    nostro_cifar10_RGB.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size1, epochs)
    nostro_cifar10_RGB.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size1, epochs)
    nostro_cifar10_RGB.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size2, epochs)
    nostro_cifar10_RGB.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size2, epochs)

    nostro_cifar10_HVS.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size1, epochs)
    nostro_cifar10_HVS.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size1, epochs)
    nostro_cifar10_HVS.train_model(conv1_out_channels1, conv2_out_channels1, fc1_neurons, kernel_size2, epochs)
    nostro_cifar10_HVS.train_model(conv1_out_channels2, conv2_out_channels2, fc1_neurons, kernel_size2, epochs)





