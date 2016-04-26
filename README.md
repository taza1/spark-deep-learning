# DL4J Spark Example

Uses [Deeplearning4j](http://deeplearning4j.org/) on top of [Apache Spark](http://spark.apache.org/) to train a [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) for digit classification, running on the [MNIST](http://yann.lecun.com/exdb/mnist/) data set.

This examples is a fork of [dl4j-spark-cdh5-examples](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples). It is set up to run on Spark local (i.e., it's runnable within your IDE).
To submit and run it on a cluster, remove the .setMaster(...) method and use Spark submit.