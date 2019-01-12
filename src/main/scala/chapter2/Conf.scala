package chapter2

import com.typesafe.config.ConfigFactory

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
