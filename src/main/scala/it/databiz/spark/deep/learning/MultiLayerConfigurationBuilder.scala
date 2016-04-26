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

package it.databiz.spark.deep.learning

import it.databiz.spark.deep.learning.Conf._
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * A dedicated MultiLayerConfiguration Builder for the MNIST dataset.
  *
  * Created by Vincibean <andrebessi00@gmail.com> on 20/03/16.
  */
object MultiLayerConfigurationBuilder extends MultiLayerConfiguration.Builder {

  def apply(): MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .regularization(true)
    .l2(0.0005)
    .learningRate(0.1)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(Updater.ADAGRAD)
    .list(6)
    .layer(0, new ConvolutionLayer.Builder(5, 5)
      .nIn(numChannels)
      .stride(1, 1)
      .nOut(20)
      .weightInit(WeightInit.XAVIER)
      .activation("relu")
      .build())
    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
      .build())
    .layer(2, new ConvolutionLayer.Builder(5, 5)
      .nIn(20)
      .nOut(50)
      .stride(2, 2)
      .weightInit(WeightInit.XAVIER)
      .activation("relu")
      .build())
    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
      .build)
    .layer(4, new DenseLayer.Builder()
      .activation("relu")
      .weightInit(WeightInit.XAVIER)
      .nOut(200)
      .build())
    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .weightInit(WeightInit.XAVIER)
      .activation("softmax")
      .build())
    .backprop(true)
    .pretrain(false)

}
