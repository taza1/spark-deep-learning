package it.databiz.spark

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.dataset.DataSet

import scala.collection.JavaConverters._

/**
  * Container object of Scala implicits to use in order to train a Convolutional Neural Network
  * from the MNIST dataset, taking advantage of Apache Spark's cluster computing.
  *
  * Created by Vincibean <andrebessi00@gmail.com> on 26/03/16.
  */
package object dl {

  /**
    * MnistDataSetIterator wrapper, loads the dataset that should be used in order to
    * train a Convolutional Neural Network on the MNIST dataset, taking advantage of
    * Apache Spark's cluster computing.
    */
  implicit class MnistDataSetIteratorOps(mnistIterator: MnistDataSetIterator) {

    /**
      * Loads the MNIST dataset to be used in the MNIST example, shuffles it, then
      * splits it into training set and test set.
      *
      * @param numForTraining the number of instances in the dataset that should be used
      *                       for the training set. It must be a number between 1 and totalNumSamples.
      * @param numSamples the number of all instances contained in the dataset.
      * @return the training set and the test set, each containing MNIST instances.
      */
    def splitDatasetAt(numForTraining: Int)(implicit numSamples: Int): (Seq[DataSet], Seq[DataSet]) = {
      require(0 < numForTraining && numForTraining < numSamples)
      val allData = mnistIterator.asScala.toSeq
      val shuffledData = scala.util.Random.shuffle(allData)
      shuffledData.splitAt(numForTraining)
    }

  }

}
