# QNN.np
Training Deep Neural Networks with Weights and Activations Quantized using Numpy.
Trained models including MNIST, SVHN, Cifar10. 
Quantization happens at each ReLu layer and also after weights were updated
Reproduced stochastic variance reduced gradient decent algorithm from paper (SVRG).

# References
The implementation structure & style is based on https://github.com/eladhoffer/convNet.tf
The implementation details (most of the functions) is based on Andrew Ng's course https://www.deeplearning.ai/
Vecterization for Conv and Pool layer is based on Agustinus Kristiadi's Blog https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/ https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
im2col.py is used in Conv and Pool layer for vecterization, provided by Stanford cs231n https://cs231n.github.io/

# USAGE
1. $ python main.py
2. $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)>
3. $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)>

SVRG:
1. $ python svrg_main.py <epoch_length(int)> <learning_rate(float)> <num_epochs(int)>
