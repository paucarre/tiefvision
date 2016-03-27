/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
  */
package core

case class Crop(left: Int, right: Int, top: Int, bottom: Int) {

  def minus(other: Crop) = Crop(left - other.left, right - other.right, top - other.top, bottom - other.bottom)

  def area = (right - left + 1) * (bottom - top + 1)

  def intersectsMoreThan50PercentWith(boundingBoxCrop: Crop) = {
    var intersectMoreThan50Percent = false
    if (intersectsWith(boundingBoxCrop)) {
      val interLeft = Math.max(left, boundingBoxCrop.left)
      val interRight = Math.min(right, boundingBoxCrop.right)
      val interTop = Math.max(top, boundingBoxCrop.top)
      val interBottom = Math.min(bottom, boundingBoxCrop.bottom)
      intersectMoreThan50Percent = ((interRight - interLeft) * (interBottom - interTop)) > area / 2
    }
    intersectMoreThan50Percent
  }

  def intersectsWith(boundingBoxCrop: Crop) =
    Math.max(left, boundingBoxCrop.left) <= Math.min(right, boundingBoxCrop.right) &&
      Math.max(top, boundingBoxCrop.top) <= Math.min(bottom, boundingBoxCrop.bottom)

}
