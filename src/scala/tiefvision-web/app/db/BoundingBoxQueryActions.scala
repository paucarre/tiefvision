/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
  */
package db

import java.util.concurrent.TimeUnit

import play.api.db.slick.{DatabaseConfigProvider, HasDatabaseConfig}
import slick.driver.H2Driver.api._
import slick.driver.JdbcProfile
import slick.jdbc.JdbcBackend
import slick.jdbc.meta.MTable
import play.api.{Play, Logger}
import scala.concurrent.{Future, Await}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

object BoundingBoxQueryActions extends App with HasDatabaseConfig[JdbcProfile] {

  lazy val dbConfig = DatabaseConfigProvider.get[JdbcProfile]("bounding_box")(Play.current)
  lazy val logger: Logger = Logger(this.getClass())
  lazy val boundingBoxTableQuery = TableQuery[BoundingBoxTable]

  def getBoundingBoxByFileName(name: String) = {
    val selectByName = boundingBoxTableQuery.filter{ boundingBoxTable =>
      boundingBoxTable.name === name
    }
    db.run(selectByName.result.headOption)
  }

  def getAllBoundingBoxes() = db.run(boundingBoxTableQuery.result)

  def insertOrUpdate(boundingBox: BoundingBox) = {
    val insertOrUpdateAction = boundingBoxTableQuery.insertOrUpdate(boundingBox)
    val insertOrUpdateResult = db.run(insertOrUpdateAction)
    insertOrUpdateResult.onFailure { case err => db: JdbcBackend#DatabaseDef
      logger.error("Unable to insert bounding box.", err)
    }
  }

}
