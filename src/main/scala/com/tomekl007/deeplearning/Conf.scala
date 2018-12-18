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

package com.tomekl007.deeplearning

import com.typesafe.config._

object Conf {

  // Load the configuration values from the default location: application.conf
  val conf = ConfigFactory.load()

  // Load the configurations to use with Spark.
  def numCores = conf.getInt("spark-config.cores")
  def batchSize = conf.getInt("spark-config.batchSize")

  // Load the hyper parameters to use in order to train the Convolutional Neural Network
  def iterations = conf.getInt("neural-network.iterations")
  def seed = conf.getInt("neural-network.seed")
  def epochs = conf.getInt("neural-network.epochs")

  // Load the dataset information.
  implicit def numSamples = conf.getInt("dataset.total")
  def numForTraining = conf.getInt("dataset.training")
  def outputNum = conf.getInt("dataset.classes")

  // Load the input images information contained in the dataset.
  def height = conf.getInt("image.height")
  def width = conf.getInt("image.width")
  def numChannels = conf.getInt("image.channels")
}
