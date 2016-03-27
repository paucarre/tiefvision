import core.Crop
import imageprocessing.ImageProcessing
import org.scalatest._

class ImageProcessingSpec extends FlatSpec with Matchers {

  "intersectsMoreThan50PercentWith" should "return true when the crop and the bounding box are the same" in {
    val crop = Crop(1, 20, 1, 20)
    val boundingBoxCrop = Crop(1, 20, 1, 20)
    val theyIntersect50Percent = crop.intersectsMoreThan50PercentWith(boundingBoxCrop)
    theyIntersect50Percent should be(true)
  }

  "intersectsMoreThan50PercentWith" should "return true when the crop and the bounding box span +50%" in {
    val crop = Crop(95, 120, 95, 120)
    val boundingBoxCrop = Crop(100, 200, 100, 200)
    val theyIntersect50Percent = crop.intersectsMoreThan50PercentWith(boundingBoxCrop)
    theyIntersect50Percent should be(true)
  }

  "intersectsMoreThan50PercentWith" should "return false when the crop and the bounding box do not intersect enough area" in {
    val crop = Crop(1, 20, 1, 20)
    val boundingBoxCrop = Crop(15, 40, 15, 40)
    val theyIntersect50Percent = crop.intersectsMoreThan50PercentWith(boundingBoxCrop)
    theyIntersect50Percent should be(false)
  }

  "intersectsMoreThan50PercentWith" should "return false when the crop and the bounding box do not intersect at all" in {
    val crop = Crop(10, 30, 10, 30)
    val boundingBoxCrop = Crop(40, 80, 40, 80)
    val theyIntersect50Percent = crop.intersectsMoreThan50PercentWith(boundingBoxCrop)
    theyIntersect50Percent should be(false)
  }

}
