# QNN.np
Training Deep Neural Networks with Weights and Activations Quantized.

This is incomplete training example for MNIST using the same model as MNIST totorial on Tensorflow (https://www.tensorflow.org/tutorials/layers). TODO: edit the following:::::::using Binary-Backpropagation algorithm as explained in "Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, on following datasets: Cifar10/100.

Note that in this folder I didn’t implemented (yet...) shift-base BN , shift-base AdaMax (instead I just use the vanilla BN and Adam). Likewise, I use deterministic binarization and I don't apply the initialization coefficients from GLorot&Bengio 2010. Finally "sparse_softmax_cross_entropy_with_logits" loss is used instead if the SquareHingeLoss.

The implementation is based on https://github.com/eladhoffer/convNet.tf but the main idea can be easily transferred to any tensorflow wrapper. I'll probably change it to keras soon. (e.g., slim,keras)
