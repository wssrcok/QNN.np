# QNN.np
Training Deep Neural Networks with Weights and Activations Quantized using Numpy.

This is incomplete training example for MNIST using the same model as MNIST totorial on Tensorflow (https://www.tensorflow.org/tutorials/layers). 
Using the Structure:
Input: 28x28 B&W image
1. Convolutional Layer #1: Applies 32 5x5 filters, with ReLU activation function
2. Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2
3. Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
4. Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
5. Dense Layer #1: 1,024 neurons
6. Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).

Quantization happens at each ReLu layer and also after weights were updated(using Gradient Decent)

# References
The implementation structure & style is based on https://github.com/eladhoffer/convNet.tf
The implementation details (most of the functions) is based on Andrew Ng's course https://www.deeplearning.ai/
Vecterization for Conv and Pool layer is based on Agustinus Kristiadi's Blog https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/ https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
im2col.py provided by Stanford cs231n https://cs231n.github.io/

# USAGE
1. $ python main.py
2. $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)>
3. $ python main.py <batch_size(int)> <learning_rate(float)> <num_epochs(int)> <quantize_bits(int)>
