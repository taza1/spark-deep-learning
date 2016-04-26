/**
  * Copyright (C) 2016  Databiz s.r.l.
  *
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  */

package it.databiz.spark.dl

import it.databiz.spark.dl.MnistConf._
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.dataset.DataSet

import scala.collection.JavaConverters._

/**
  * Loads the dataset that should be used in order to train a Convolutional Neural Network
  * on the MNIST dataset, taking advantage of Apache Spark's cluster computing.
  *
  * Created by Vincibean <andrebessi00@gmail.com> on 20/03/16.
  */
object MnistDataSet {

  /**
    * Loads the MNIST dataset to be used in the MNIST example, shuffles it, then
    * splits it into training set and test set.
    *
    * @param numForTraining the number of instances in the dataset that should be used
    *                       for the training set. It must be a number between 1 and totalNumSamples.
    * @param totalNumSamples the number of all instances contained in the dataset.
    * @return the training set and the test set, each containing MNIST instances.
    */
  def dataset(numForTraining: Int, totalNumSamples: Int = 60000): (Seq[DataSet], Seq[DataSet]) = {
    require(0 < numForTraining && numForTraining < totalNumSamples)
    val allData = new MnistDataSetIterator(1, numSamples, true).asScala.toSeq
    val shuffledData = scala.util.Random.shuffle(allData)
    shuffledData.splitAt(numForTraining)
  }

}
