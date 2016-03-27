/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
  */
package db
import slick.driver.H2Driver.api._

object Dataset extends Enumeration {
  type Dataset = Value
  val TrainingSet = Value("TRAIN")
  val TestSet = Value("TEST")
  val UndefinedSet = Value("UNKNOWN")

  implicit val fDatasetMapper = MappedColumnType.base[Dataset, String](
    { e => e.toString }, { s => Dataset.withName(s) }
  )

}
