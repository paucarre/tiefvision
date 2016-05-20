/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
  */
package db

import slick.driver.H2Driver.api._
import db.Dataset._

class SimilarityTable(tag: Tag) extends Table[Similarity](tag, "SIMILARITY") {

  def reference = column[String]("REFERENCE")

  def positive = column[String]("POSITIVE")

  def negative = column[String]("NEGATIVE")

  def dataset = column[Dataset]("DATASET")

  def pk = primaryKey("pk", (reference, positive, negative))

  def * = (reference, positive, negative, dataset) <>(Similarity.tupled, Similarity.unapply)

}

case class Similarity(reference: String, positive: String, negative: String, dataset: Dataset) {

  def isFlipped = reference == positive
  def isTest = dataset == Dataset.TestSet

}
