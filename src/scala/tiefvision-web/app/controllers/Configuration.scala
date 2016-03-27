package controllers

import play.api.Play

object Configuration {

  lazy val HomeFolder = sys.env("TIEFVISION_HOME")
  val CropSize = 224
  val NumSamples = 5
  val ClassificationFolder = "resources/classification-images/crops"
  val BoundingBoxesFolder = "resources/bounding-boxes"
  val ScaledImagesFolder = "resources/bounding-boxes/scaled-images"
  val CropImagesFolder = "resources/bounding-boxes/crops"
  val BackgroundCropImagesFolder = "resources/bounding-boxes/background-crops"
  val DbImagesFolder = "resources/dresses-db/master"
  val scaleLevels = Seq(2, 3)
  val testPercentage = 0.05
  
}
