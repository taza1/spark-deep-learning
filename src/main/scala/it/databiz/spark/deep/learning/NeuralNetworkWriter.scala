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

import java.io.{DataOutputStream, File}
import java.nio.file.{Files, Paths}

import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

import scala.util.Try

/**
  * Utility object, should be used to save a Convolutional Neural Network's configurations on disk.
  *
  * Created by Vincibean <andrebessi00@gmail.com> on 26/04/16.
  */
object NeuralNetworkWriter {

  /**
    * Saves a Convolutional Neural Network's configurations and coefficients on disk.
    *
    * @param neuralNetwork the Convolutional Neural Network whose configurations
    *                      and coefficients should be saved on disk.
    * @return a Try monad indicating if the computation resulted in an Exception or not.
    */
  def write(neuralNetwork: MultiLayerNetwork): Try[Unit] = Try {
    //Write the network parameters:
    val output = new DataOutputStream(Files.newOutputStream(Paths.get("coefficients.bin")))
    Nd4j.write(neuralNetwork.params(), output)

    //Write the network configuration:
    FileUtils.write(new File("conf.json"), neuralNetwork.getLayerWiseConfigurations.toJson)
  }

}
