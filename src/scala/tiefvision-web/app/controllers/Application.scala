/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
  */
package controllers

import _root_.db._
import core.{DatabaseProcessing, ImageProcessing}
import ImageProcessing._
import play.api.mvc._
import scala.concurrent.ExecutionContext.Implicits.global

class Application extends Controller {

  lazy val ImagesGrouped = Images.toList.grouped(20).toList

  def index = editBoundingBox(randomImage)

  def similarityGallery(page: Int = 1, pageGroup: Int = 1) = Action {
    Ok(views.html.similarityGallery(ImagesGrouped(((pageGroup - 1) * 20) + (page - 1)), page, pageGroup, ImagesGrouped.size / 20))
  }

  def similarityFinderUploadForm() = Action {
    Ok(views.html.similarityFinderUploadForm())
  }

  def similarityFinderUploadResults(image: String) = Action {
    val imageSearchResult = findSimilarImages(image, findSimilarImagesFromFileFolder,
      Some(s"${Configuration.HomeFolder}/${Configuration.UploadedImagesFolder}"))
    Ok(views.html.similarityFinder(imageSearchResult, "uploaded_dresses_db", "uploaded_bboxes_db"))
  }

  def upload = Action(parse.multipartFormData) { request =>
    request.body.file("picture").map { picture =>
      import java.io.File
      val filename = picture.filename
      val contentType = picture.contentType
      picture.ref.moveTo(new File(s"${Configuration.HomeFolder}/resources/dresses-db/uploaded/master/${filename}"), true)
      Redirect(routes.Application.similarityFinderUploadResults(filename))
    }.getOrElse {
      Redirect(routes.Application.index).flashing(
        "error" -> "Missing file"
      )
    }
  }

  def similarityFinder() = Action {
    val imageSearchResult = findSimilarImages(randomSmilarityImage, findSimilarImagesFromDbFolder)
    Ok(views.html.similarityFinder(imageSearchResult, "dresses_db", "bboxes_db"))
  }

  def similarityFinderFor(image: String) = Action {
    val imageSearchResult = findSimilarImages(image, findSimilarImagesFromDbFolder)
    Ok(views.html.similarityFinder(imageSearchResult, "dresses_db", "bboxes_db"))
  }

  def similarityEditor() = similarityEditorFor(randomSmilarityImage)

  def similarityEditorFor(image: String) = Action.async {
    val imageSearchResult = findSimilarImages(image, findSimilarImagesFromDbFolder)
    SimilarityQueryActions.getSimilarityByReference(image).map { similarity =>
      Ok(views.html.similarityEditor(imageSearchResult, "dresses_db", "bboxes_db",
        similarity.map(_.positive), similarity.map(_.negative)))
    }
  }

  def saveBoundingBox(name: String, left: Int, right: Int, top: Int, bottom: Int, width: Int, height: Int) = Action {
    BoundingBoxQueryActions.insertOrUpdate(BoundingBox(name = name, top = top,
      left = left, bottom = bottom, right = right, width = width, height = height, dataset = Dataset.UndefinedSet))
    Redirect(routes.Application.index)
  }

  def saveSimilarity(reference: String, positive: String, negative: String) = Action {
    SimilarityQueryActions.insertOrUpdate(Similarity(reference = reference, positive = positive,
      negative = negative, dataset = Dataset.UndefinedSet))
    Redirect(routes.Application.similarityEditor())
  }

  def editBoundingBox(name: String) = Action.async {
    val boundingBoxFutOpt = BoundingBoxQueryActions.getBoundingBoxByFileName(name)
    boundingBoxFutOpt.map { boundingBoxOpt =>
      boundingBoxOpt.fold {
        Ok(views.html.editBoundingBox(name))
      } { boundingBox =>
        Ok(views.html.editBoundingBox(name, Some(boundingBox)))
      }
    }
  }

  def generateSimilarityTrainAndTestFiles() = Action.async {
    DatabaseProcessing.generateSimilarityTestAndTrainFiles().map { generated =>
      Ok("Similarity Test And Train Files Generated.")
    }
  }

  def generateBoundingBoxesCrops() = Action.async {
    DatabaseProcessing.generateBoundingBoxDatabaseImages(true).map { generated =>
      Ok("Bounding Boxes Crops Generated.")
    }
  }

  def generateBoundingBoxTrainAndTestFiles() = Action.async {
    DatabaseProcessing.generateBoundingBoxTrainAndTestFiles().map { trainAndTestFiles =>
      Ok("Bounding Boxe Train and Test Files Generated.")
    }
  }

  def generateClassificationTrainAndTestFiles() = Action.async {
    DatabaseProcessing.generateClassificationTrainAndTestFiles().map { trainAndTestFiles =>
      Ok("Classification Train and Test Files Generated.")
    }
  }


}
