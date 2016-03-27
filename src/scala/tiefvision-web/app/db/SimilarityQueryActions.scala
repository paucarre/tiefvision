/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
  */
package db

import play.api.db.slick.{DatabaseConfigProvider, HasDatabaseConfig}
import play.api.{Logger, Play}
import slick.driver.H2Driver.api._
import slick.driver.JdbcProfile
import slick.jdbc.JdbcBackend

import scala.concurrent.ExecutionContext.Implicits.global

object SimilarityQueryActions extends App with HasDatabaseConfig[JdbcProfile] {

  lazy val dbConfig = DatabaseConfigProvider.get[JdbcProfile]("bounding_box")(Play.current)
  lazy val logger: Logger = Logger(this.getClass())
  lazy val similarityTableQuery = TableQuery[SimilarityTable]

  def getSimilarityByReference(name: String) = {
    val selectByName = similarityTableQuery.filter{ similarityTable =>
      similarityTable.reference === name
    }
    db.run(selectByName.result.headOption)
  }

  def getUnflippedSimilarities() = {
    val unflippedQuery = similarityTableQuery.filter{ similarityTable =>
      similarityTable.reference =!= similarityTable.positive
    }
    db.run(unflippedQuery.result)
  }

  def getFlippedSimilarities() = {
    val flippedQuery = similarityTableQuery.filter{ similarityTable =>
      similarityTable.reference === similarityTable.positive
    }
    db.run(flippedQuery.result)
  }

  def getAllSimilarities() = db.run(similarityTableQuery.result)

  def insertOrUpdate(similarity: Similarity) = {
    val insertOrUpdateAction = similarityTableQuery.insertOrUpdate(similarity)
    val insertOrUpdateResult = db.run(insertOrUpdateAction)
    insertOrUpdateResult.onFailure { case err => db: JdbcBackend#DatabaseDef
      logger.error("Unable to insert similarity.", err)
    }
  }

}
