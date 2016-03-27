/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
  */
package db

import core.Crop
import db.Dataset._
import slick.driver.H2Driver.api._


class BoundingBoxTable(tag: Tag) extends Table[BoundingBox](tag, "BOUNDING_BOX") {

  def name = column[String]("FILE_NAME", O.PrimaryKey)

  def top = column[Int]("TOP")

  def left = column[Int]("LEFT")

  def bottom = column[Int]("BOTTOM")

  def right = column[Int]("RIGHT")

  def width = column[Int]("WIDTH")

  def height = column[Int]("HEIGHT")

  def dataset = column[Dataset]("DATASET")

  def * = (name, top, left, bottom, right, width, height, dataset) <>(BoundingBox.tupled, BoundingBox.unapply)

}

case class BoundingBox(name: String, top: Int,
                       left: Int, bottom: Int, right: Int, width: Int, height: Int, dataset: Dataset) {

  def toCrop = Crop(left, right, top, bottom)

  def div(denominator: Double) = BoundingBox(
    name   = name,
    top    = (top.toDouble    / denominator).ceil.toInt,
    left   = (left.toDouble   / denominator).ceil.toInt,
    bottom = (bottom.toDouble / denominator).ceil.toInt,
    right  = (right.toDouble  / denominator).ceil.toInt,
    width  = (width.toDouble  / denominator).ceil.toInt,
    height = (height.toDouble / denominator).ceil.toInt,
    dataset = dataset)
}
