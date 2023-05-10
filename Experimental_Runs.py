from ModelFunctions import *
import csv
import timeit

#activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']


#testing different neurons
def run1_neurons():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 1
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(5):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[1], neurons[i1], csv_path, layers,
                                     activations[0])
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[2])
        tmpstore.append(batch_size[1])
        tmpstore.append(epochs[1])
        tmpstore.append(neurons[i1])
        tmpstore.append(csv_path)
        tmpstore.append(layers)
        tmpstore.append(activations[0])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)


#testing different learning rates
def run1_lr():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 1
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(3):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[i1], batch_size[1], epochs[1], neurons[2], csv_path, layers,
                                     activations[0])
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[i1])
        tmpstore.append(batch_size[1])
        tmpstore.append(epochs[1])
        tmpstore.append(neurons[2])
        tmpstore.append(csv_path)
        tmpstore.append(layers)
        tmpstore.append(activations[0])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)


#testing different batchsizes
def run1_bs():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 1
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(4):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[2], batch_size[i1], epochs[1], neurons[2], csv_path, layers,
                                     activations[0])
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[2])
        tmpstore.append(batch_size[i1])
        tmpstore.append(epochs[1])
        tmpstore.append(neurons[2])
        tmpstore.append(csv_path)
        tmpstore.append(layers)
        tmpstore.append(activations[0])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)

#testing different epochs
def run1_epoch():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 1
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(4):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[i1], neurons[2], csv_path, layers,
                                     activations[0])
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[2])
        tmpstore.append(batch_size[1])
        tmpstore.append(epochs[i1])
        tmpstore.append(neurons[2])
        tmpstore.append(csv_path)
        tmpstore.append(layers)
        tmpstore.append(activations[0])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)

#testing different files
def run1_files():
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = ['data_reformatted1.csv', 'data_reformatted2.csv', 'data_reformatted3.csv', 'data_reformatted4.csv', 'data_reformatted5.csv']
    layers = 1
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(5):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[1], neurons[2], csv_path[i1], layers,
                                     activations[0])
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[2])
        tmpstore.append(batch_size[1])
        tmpstore.append(epochs[1])
        tmpstore.append(neurons[2])
        tmpstore.append(csv_path[i1])
        tmpstore.append(layers)
        tmpstore.append(activations[0])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)


#testing different layers
def run1_layers():
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'data_reformatted1.csv'
    layers = [1,2,3,4]
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(4):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[1], [neurons[2]]*i1, csv_path, layers[i1],
                                     [activations[0]] * i1)
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[2])
        tmpstore.append(batch_size[1])
        tmpstore.append(epochs[1])
        tmpstore.append([neurons[2]]*i1)
        tmpstore.append(csv_path)
        tmpstore.append(layers[i1])
        tmpstore.append([activations[0]])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)


#testing different activations
def run1_activations():
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'data_reformatted1.csv'
    layers = 1
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(8):
        tmpstore = []
        start_time = timeit.default_timer()
        model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[1], neurons[2], csv_path, layers,
                                     activations[i1])
        end_time = timeit.default_timer() - start_time
        tmpstore.append(learning_rate[2])
        tmpstore.append(batch_size[1])
        tmpstore.append(epochs[1])
        tmpstore.append(neurons[2])
        tmpstore.append(csv_path)
        tmpstore.append(layers)
        tmpstore.append([activations[i1]])
        tmpstore.append(r2)
        tmpstore.append(meanSq)
        tmpstore.append(end_time)
        model_info.append(tmpstore)
        print(tmpstore)

#testing different neuron counts and layer counts
def run2_neur_lay():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = ['relu', 'sigmoid', 'selu', 'softplus', 'softsign', 'tanh', 'elu', 'exponential']
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(1, 6):

        for j1 in range(5):
            tmpstore = []
            start_time = timeit.default_timer()
            model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[1], neurons[j1:j1+i1], csv_path, i1, [activations[0]]*i1)
            end_time = timeit.default_timer() - start_time
            tmpstore.append(learning_rate[2])
            tmpstore.append(batch_size[1])
            tmpstore.append(epochs[1])
            tmpstore.append(neurons[j1:j1+i1])
            tmpstore.append(csv_path)
            tmpstore.append(i1)
            tmpstore.append([activations[0]*i1])
            tmpstore.append(r2)
            tmpstore.append(meanSq)
            tmpstore.append(end_time)
            model_info.append(tmpstore)
            print(tmpstore)



#testing different neuron counts and activation functions
def run2_neur_act():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = [['relu', 'relu', 'relu'], ['sigmoid', 'sigmoid', 'sigmoid'], ['softsign', 'softsign', 'softsign'], ['relu', 'sigmoid', 'softsign'], ['relu', 'sigmoid', 'relu']]
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(5):

        for j1 in range(5):
            tmpstore = []
            start_time = timeit.default_timer()
            model, meanSq, r2 = train_nn(learning_rate[2], batch_size[1], epochs[1], [neurons[j1]]*layers, csv_path, layers, activations[i1])
            end_time = timeit.default_timer() - start_time
            tmpstore.append(learning_rate[2])
            tmpstore.append(batch_size[1])
            tmpstore.append(epochs[1])
            tmpstore.append([neurons[j1]*layers])
            tmpstore.append(csv_path)
            tmpstore.append(layers)
            tmpstore.append(activations[i1])
            tmpstore.append(r2)
            tmpstore.append(meanSq)
            tmpstore.append(end_time)
            model_info.append(tmpstore)
            print(tmpstore)


#testing different learning rates and epochs
def run2_lr_ep():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = [['relu', 'relu', 'relu'], ['sigmoid', 'sigmoid', 'sigmoid'], ['softsign', 'softsign', 'softsign'], ['relu', 'sigmoid', 'softsign'], ['relu', 'sigmoid', 'relu']]
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(3):

        for j1 in range(5):
            tmpstore = []
            start_time = timeit.default_timer()
            model, meanSq, r2 = train_nn(learning_rate[i1], batch_size[1], epochs[j1], [neurons[1]]*layers, csv_path, layers, activations[0])
            end_time = timeit.default_timer() - start_time
            tmpstore.append(learning_rate[i1])
            tmpstore.append(batch_size[1])
            tmpstore.append(epochs[j1])
            tmpstore.append([neurons[1]*layers])
            tmpstore.append(csv_path)
            tmpstore.append(layers)
            tmpstore.append(activations[0])
            tmpstore.append(r2)
            tmpstore.append(meanSq)
            tmpstore.append(end_time)
            model_info.append(tmpstore)
            print(tmpstore)


#testing different epochs and batchsizes
def run2_ep_bs():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = [['relu', 'relu', 'relu'], ['sigmoid', 'sigmoid', 'sigmoid'], ['softsign', 'softsign', 'softsign'], ['relu', 'sigmoid', 'softsign'], ['relu', 'sigmoid', 'relu']]
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(4):

        for j1 in range(4):
            tmpstore = []
            start_time = timeit.default_timer()
            model, meanSq, r2 = train_nn(learning_rate[2], batch_size[i1], epochs[j1], [neurons[2]]*layers, csv_path, layers, activations[0])
            end_time = timeit.default_timer() - start_time
            tmpstore.append(learning_rate[2])
            tmpstore.append(batch_size[i1])
            tmpstore.append(epochs[j1])
            tmpstore.append([neurons[2]*layers])
            tmpstore.append(csv_path)
            tmpstore.append(layers)
            tmpstore.append(activations[0])
            tmpstore.append(r2)
            tmpstore.append(meanSq)
            tmpstore.append(end_time)
            model_info.append(tmpstore)
            print(tmpstore)


#testing different kernel sizes and filter count
def run2_ker_fil_nopool():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    kernels = [20, 50, 100, 200, 500]
    filters = [1, 10, 50, 100, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = [['relu', 'relu', 'relu'], ['sigmoid', 'sigmoid', 'sigmoid'], ['softsign', 'softsign', 'softsign'], ['relu', 'sigmoid', 'softsign'], ['relu', 'sigmoid', 'relu']]
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "kernel", "filters", "r2", "mse", "runtime"]
    for i1 in range(5):

        for j1 in range(6):
            tmpstore = []
            start_time = timeit.default_timer()
            model, meanSq, r2 = train_cnn_noPool(learning_rate, batch_size, epochs, neurons, csv_path, layers, activations, kernels[i1], filters[j1])
            end_time = timeit.default_timer() - start_time
            tmpstore.append(learning_rate[2])
            tmpstore.append(batch_size[1])
            tmpstore.append(epochs[1])
            tmpstore.append([neurons[2]*layers])
            tmpstore.append(csv_path)
            tmpstore.append(layers)
            tmpstore.append(activations[0])
            tmpstore.append(kernels[i1])
            tmpstore.append(filters[j1])
            tmpstore.append(r2)
            tmpstore.append(meanSq)
            tmpstore.append(end_time)
            model_info.append(tmpstore)
            print(tmpstore)

#testing different kernel sizes and filter count
def run2_ker_fil():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    kernels = [20, 50, 100, 200, 500]
    filters = [1, 10, 50, 100, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = [['relu', 'relu', 'relu'], ['sigmoid', 'sigmoid', 'sigmoid'], ['softsign', 'softsign', 'softsign'], ['relu', 'sigmoid', 'softsign'], ['relu', 'sigmoid', 'relu']]
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "kernel", "filters", "r2", "mse", "runtime"]
    for i1 in range(5):

        for j1 in range(6):
            tmpstore = []
            start_time = timeit.default_timer()
            model, meanSq, r2 = train_cnn(learning_rate, batch_size, epochs, neurons, csv_path, layers, activations, kernels[i1], filters[j1])
            end_time = timeit.default_timer() - start_time
            tmpstore.append(learning_rate[2])
            tmpstore.append(batch_size[1])
            tmpstore.append(epochs[1])
            tmpstore.append([neurons[2]*layers])
            tmpstore.append(csv_path)
            tmpstore.append(layers)
            tmpstore.append(activations[0])
            tmpstore.append(kernels[i1])
            tmpstore.append(filters[j1])
            tmpstore.append(r2)
            tmpstore.append(meanSq)
            tmpstore.append(end_time)
            model_info.append(tmpstore)
            print(tmpstore)

#testing different learning rate, epoch, and batchsizes
def run3_lr_ep_bs():
    #use only 1 of the 5 files
    model_info = []
    learning_rate = [.001, .01, .1]
    batch_size = [10, 20, 50, 100]
    epochs = [50, 100, 200, 500]
    neurons = [50, 100, 200, 300, 500]
    csv_path = 'csv_name'
    layers = 3
    activations = [['relu', 'relu', 'relu'], ['sigmoid', 'sigmoid', 'sigmoid'], ['softsign', 'softsign', 'softsign'], ['relu', 'sigmoid', 'softsign'], ['relu', 'sigmoid', 'relu']]
    model_info = ["learning rate", "batch size", "epochs", "neurons", "file", "layers", "activations", "r2", "mse", "runtime"]
    for i1 in range(3):
        for j1 in range(4):
            for k1 in range(4):
                tmpstore = []
                start_time = timeit.default_timer()
                model, meanSq, r2 = train_nn(learning_rate[i1], batch_size[j1], epochs[k1], [neurons[2]]*layers, csv_path, layers, activations[0])
                end_time = timeit.default_timer() - start_time
                tmpstore.append(learning_rate[i1])
                tmpstore.append(batch_size[j1])
                tmpstore.append(epochs[k1])
                tmpstore.append([neurons[2]*layers])
                tmpstore.append(csv_path)
                tmpstore.append(layers)
                tmpstore.append(activations[0])
                tmpstore.append(r2)
                tmpstore.append(meanSq)
                tmpstore.append(end_time)
                model_info.append(tmpstore)
                print(tmpstore)





