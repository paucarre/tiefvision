/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
  */
package core

import java.io.{BufferedWriter, File, FileOutputStream, OutputStreamWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import controllers.Configuration
import core.ImageProcessing.SimilarityFinder
import db.Dataset._
import db.{Dataset, _}

import scala.concurrent.ExecutionContext.Implicits.global

object DatabaseProcessing {

  private lazy val Images = new File(s"${Configuration.HomeFolder}/resources/dresses-db/master").listFiles().map(_.getName)
  private lazy val SimilarityImages = new File(s"${Configuration.HomeFolder}/src/torch/data/db/similarity/img-enc-cnn").listFiles().map(_.getName)

  private val random = scala.util.Random

  def randomImage = Images(random.nextInt(Images.length))

  def getUnsupervisedAccuracy() = {
    SimilarityQueryActions.getAllSimilarities().map { allSimilarities =>
      allSimilarities.foldLeft(Accuracy())((accuracy, similarity) => {
        val imageToDist = ImageProcessing.findSimilarImages(similarity.reference,
          SimilarityFinder.getSimilarityFinder(false)).distanceToSimilarImages.map(e => e._2 -> e._1).toMap
        val isCorrect = for {
          positiveDist <- imageToDist.get(similarity.positive)
          negativeDist <- imageToDist.get(similarity.negative)
        } yield {
          positiveDist < negativeDist
        }
        accuracy.update(isCorrect = isCorrect.getOrElse(false), isFlipped = similarity.isFlipped, isTest = similarity.isTest)
      })
    }
  }

  def generateSimilarityTestAndTrainFiles() = {
    generateFlippedSimilarityDbForUnflippedReferenceImages
    generateRandomFlippedSimilarityDb
    assignSimilaritiesToTestAndTrainDataset
    generateTextTrainAndTestFiles
  }

  def generateTextTrainAndTestFiles() = {
    def text(similarity: Similarity) = s"${similarity.reference},${similarity.positive},${similarity.negative}"
    val similaritiesFut = SimilarityQueryActions.getAllSimilarities()
    similaritiesFut.map { similarities =>
      val shuffledSimilarities = random.shuffle(similarities)
      val testSimilaritiesDbAsString = shuffledSimilarities.filter(_.dataset == Dataset.TestSet).map(text(_)).mkString("\n")
      val trainSimilaritiesDbAsString = shuffledSimilarities.filter(_.dataset == Dataset.TrainingSet).map(text(_)).mkString("\n")
      Files.write(Paths.get(s"${Configuration.HomeFolder}/resources/dresses-db/similarity-db-train"), trainSimilaritiesDbAsString.getBytes(StandardCharsets.UTF_8))
      Files.write(Paths.get(s"${Configuration.HomeFolder}/resources/dresses-db/similarity-db-test"), testSimilaritiesDbAsString.getBytes(StandardCharsets.UTF_8))
    }
  }

  def assignSimilaritiesToTestAndTrainDataset() = {
    for {
      unflippedSimilarities <- SimilarityQueryActions.getUnflippedSimilarities()
      flippedSimilarities <- SimilarityQueryActions.getFlippedSimilarities()
    } yield {
      assignSimilaritiesToTestAndTrainSimilarities(unflippedSimilarities)
      assignSimilaritiesToTestAndTrainSimilarities(flippedSimilarities)
    }
  }

  private def assignSimilaritiesToTestAndTrainSimilarities(similarities: Seq[Similarity]) = {
    val undefined = similarities.filter(_.dataset == Dataset.UndefinedSet)
    val test = similarities.filter(_.dataset == Dataset.TestSet)
    val testToAddCount = (Configuration.testPercentage * similarities.size.toDouble) - test.size.toDouble
    val testToAdd = random.shuffle(undefined).take(testToAddCount.toInt)
    testToAdd.foreach { test =>
      SimilarityQueryActions.insertOrUpdate(test.copy(dataset = Dataset.TestSet))
    }
    val trainToAdd = undefined.toSet &~ testToAdd.toSet
    trainToAdd.foreach { train =>
      SimilarityQueryActions.insertOrUpdate(train.copy(dataset = Dataset.TrainingSet))
    }
  }

  def generateRandomFlippedSimilarityDb() =
    for {
      unflippedSimilarities <- SimilarityQueryActions.getUnflippedSimilarities()
      flippedSimilarities <- SimilarityQueryActions.getFlippedSimilarities()
    } yield {
      val currentRandomFlippedSimilarities = flippedSimilarities.map(_.reference).toSet &~ (unflippedSimilarities.map(_.reference).toSet)
      val randomFlippedSimilaritiesCountToAdd = unflippedSimilarities.map(_.reference).size - currentRandomFlippedSimilarities.size
      val possibleFlippedSimilaritiesToAdd = SimilarityImages.toSet.&~(unflippedSimilarities.map(_.reference).toSet.union(currentRandomFlippedSimilarities))
      val flippedSimilaritiesToAdd = random.shuffle(possibleFlippedSimilaritiesToAdd).take(randomFlippedSimilaritiesCountToAdd)
      flippedSimilaritiesToAdd.foreach { flippedSimilarityToAdd =>
        addFlippedSimilarity(flippedSimilarityToAdd)
      }
    }

  private def addFlippedSimilarity(flippedSimilarityToAdd: String) = {
    val closestImageOpt = ImageProcessing.findSimilarImages(flippedSimilarityToAdd,
      SimilarityFinder.getSimilarityFinder(true)).distanceToSimilarImages.headOption
    closestImageOpt.map { closestImage =>
      val closestImageName = closestImage._2
      SimilarityQueryActions.insertOrUpdate(
        Similarity(
          reference = flippedSimilarityToAdd,
          positive = flippedSimilarityToAdd,
          negative = closestImageName,
          dataset = Dataset.UndefinedSet
        )
      )
    }
  }

  def generateFlippedSimilarityDbForUnflippedReferenceImages() =
    for {
      unflippedSimilarities <- SimilarityQueryActions.getUnflippedSimilarities()
      flippedSimilarities <- SimilarityQueryActions.getFlippedSimilarities()
    } yield {
      val flippedSimilaritiesToAdd = unflippedSimilarities.map(_.reference).toSet.&~(flippedSimilarities.map(_.reference).toSet)
      flippedSimilaritiesToAdd.foreach { flippedSimilarityToAdd =>
        addFlippedSimilarity(flippedSimilarityToAdd)
      }
    }

  def generateBoundingBoxTrainAndTestFiles() =
    generateBoundingBoxDatabaseImages(true) flatMap (imageUnit =>
      setDatasetToUnknownDatasets() flatMap (datasetUnit =>
        generateBoundingBoxTrainAndTestSetFiles(true)
        )
      )

  def generateClassificationTrainAndTestFiles() =
    generateBoundingBoxDatabaseImages(false) flatMap (imageUnit =>
      generateBackgroundBoundingBoxDatabaseImages(false) flatMap (backgroundImageUnit =>
        setDatasetToUnknownDatasets() flatMap (datasetUnit =>
          generateClassificationTrainAndTestSetFiles(false)
          )
        )
      )

  def generateBoundingBoxTrainAndTestSetFiles(extendedBoundingBox: Boolean) = {
    val boundingBoxesSeqFut = BoundingBoxQueryActions.getAllBoundingBoxes()
    boundingBoxesSeqFut.map { boundingBoxesSeq =>
      generateBoundingBoxDatasetFile(boundingBoxesSeq, Dataset.TrainingSet, extendedBoundingBox)
      generateBoundingBoxDatasetFile(boundingBoxesSeq, Dataset.TestSet, extendedBoundingBox)
    }
  }

  def generateClassificationTrainAndTestSetFiles(extendBoundingBox: Boolean) = {
    val boundingBoxesSeqFut = BoundingBoxQueryActions.getAllBoundingBoxes()
    boundingBoxesSeqFut.map { boundingBoxesSeq =>
      generateClassificationDatasetFile(boundingBoxesSeq, Dataset.TrainingSet, extendBoundingBox)
      generateClassificationDatasetFile(boundingBoxesSeq, Dataset.TestSet, extendBoundingBox)
      generateClassificationBackgroundDatasetFile(boundingBoxesSeq, Dataset.TrainingSet, extendBoundingBox)
      generateClassificationBackgroundDatasetFile(boundingBoxesSeq, Dataset.TestSet, extendBoundingBox)
    }
  }

  def generateClassificationBackgroundDatasetFile(boundingBoxes: Seq[BoundingBox], dataset: Dataset, extendBoundingBox: Boolean) = {
    val cropsFolderName = s"${Configuration.HomeFolder}/${Configuration.BackgroundCropImagesFolder}/${ImageProcessing.boundingBoxTypeFolder(extendBoundingBox)}"
    val cropsDirectoryFiles = new File(cropsFolderName).listFiles()
    val boundingBoxesInDataset = boundingBoxes.filter(_.dataset == dataset)
    val content = cropsDirectoryFiles.toSet
      .filter(file => boundingBoxesInDataset.find(boundingBox => file.getName startsWith boundingBox.name).isDefined)
      .map(file => s"$cropsFolderName/${file.getName}").mkString("\n")

    val writer = new BufferedWriter(new OutputStreamWriter(
      new FileOutputStream(s"${Configuration.HomeFolder}/${Configuration.ClassificationFolder}/13-${dataset.toString.toLowerCase}.txt"), "utf-8"))
    writer.write(content)
    writer.close()
  }

  def generateClassificationDatasetFile(boundingBoxes: Seq[BoundingBox], dataset: Dataset, extendBoundingBox: Boolean) = {
    val cropsFolderName = s"${Configuration.HomeFolder}/${Configuration.CropImagesFolder}/${ImageProcessing.boundingBoxTypeFolder(extendBoundingBox)}"
    val cropsDirectoryFiles = new File(cropsFolderName).listFiles()
    val boundingBoxesInDataset = boundingBoxes.filter(_.dataset == dataset)
    val content = cropsDirectoryFiles.toSet
      .filter(file => boundingBoxesInDataset.find(boundingBox => file.getName startsWith boundingBox.name).isDefined)
      .map(file => s"$cropsFolderName/${file.getName}").mkString("\n")

    val writer = new BufferedWriter(new OutputStreamWriter(
      new FileOutputStream(s"${Configuration.HomeFolder}/${Configuration.ClassificationFolder}/1-${dataset.toString.toLowerCase}.txt"), "utf-8"))
    writer.write(content)
    writer.close()
  }

  def generateBoundingBoxDatasetFile(boundingBoxes: Seq[BoundingBox], dataset: Dataset, extendBoundingBox: Boolean) = {
    val cropsFolderName = s"${Configuration.HomeFolder}/${Configuration.CropImagesFolder}/${ImageProcessing.boundingBoxTypeFolder(extendBoundingBox)}"
    val cropsDirectoryFiles = new File(cropsFolderName).listFiles()
    val boundingBoxesInDataset = boundingBoxes.filter(_.dataset == dataset)
    val content = cropsDirectoryFiles.toSet
      .filter(file => boundingBoxesInDataset.find(boundingBox => file.getName startsWith boundingBox.name).isDefined)
      .map(file => s"$cropsFolderName/${file.getName}").mkString("\n")

    val writer = new BufferedWriter(new OutputStreamWriter(
      new FileOutputStream(s"${Configuration.HomeFolder}/${Configuration.BoundingBoxesFolder}/${ImageProcessing.boundingBoxTypeFolder(extendBoundingBox)}${dataset.toString}.txt"), "utf-8"))
    writer.write(content)
    writer.close()
  }

  def setDatasetToUnknownDatasets() = {
    val boundingBoxesSeqFut = BoundingBoxQueryActions.getAllBoundingBoxes()
    boundingBoxesSeqFut.map { boundingBoxesSeq =>
      val testCount = boundingBoxesSeq.filter(_.dataset == Dataset.TestSet).size.toDouble
      val undefinedBoundingBoxes = boundingBoxesSeq.filter(_.dataset == Dataset.UndefinedSet)
      val testSamplesCountToAdd = ((boundingBoxesSeq.size.toDouble * Configuration.testPercentage) - testCount).toInt
      val testSamplesToAdd = undefinedBoundingBoxes.take(testSamplesCountToAdd)
      val trainSamplesToAdd = undefinedBoundingBoxes.toSet &~ testSamplesToAdd.toSet
      testSamplesToAdd.foreach { testSampleToAdd =>
        BoundingBoxQueryActions.insertOrUpdate(testSampleToAdd.copy(dataset = Dataset.TestSet))
      }
      trainSamplesToAdd.foreach { trainSampleToAdd =>
        BoundingBoxQueryActions.insertOrUpdate(trainSampleToAdd.copy(dataset = Dataset.TrainingSet))
      }
    }
  }

  def generateBoundingBoxDatabaseImages(extendBoundingBox: Boolean) = {
    val boundingBoxesSeqFut = BoundingBoxQueryActions.getAllBoundingBoxes()
    boundingBoxesSeqFut.map { boundingBoxesSeq =>
      boundingBoxesSeq.foreach { boundingBox =>
        if (!cropsGenerated(boundingBox)) {
          Configuration.scaleLevels.foreach { scaleLevel =>
            val scale = boundingBox.width.toDouble / (Configuration.CropSize.toDouble * scaleLevel.toDouble)
            val scaledBoundingBox = boundingBox div scale
            ImageProcessing.saveImageScaled(scaledBoundingBox, scaleLevel)
            ImageProcessing.generateCrops(scaledBoundingBox, scaleLevel, extendBoundingBox)
          }
        }
      }
    }
  }

  def generateBackgroundBoundingBoxDatabaseImages(extendBoundingBox: Boolean) = {
    val boundingBoxesSeqFut = BoundingBoxQueryActions.getAllBoundingBoxes()
    boundingBoxesSeqFut.map { boundingBoxesSeq =>
      boundingBoxesSeq.foreach { boundingBox =>
        if (!backgroundCropsGenerated(boundingBox)) {
          Configuration.scaleLevels.foreach { scaleLevel =>
            val scale = boundingBox.width.toDouble / (Configuration.CropSize.toDouble * scaleLevel.toDouble)
            val scaledBoundingBox = boundingBox div scale
            ImageProcessing.saveImageScaled(scaledBoundingBox, scaleLevel)
            ImageProcessing.generateBackgroundCrops(scaledBoundingBox, scaleLevel, extendBoundingBox)
          }
        }
      }
    }
  }

  def cropsGenerated(scaledBoundingBox: BoundingBox): Boolean = {
    val cropsFolderName = s"${Configuration.HomeFolder}/${Configuration.CropImagesFolder}"
    val cropsDirectoryFiles = new File(cropsFolderName).listFiles()
    cropsDirectoryFiles.foldLeft(false)((exists, file) => exists || file.getName.startsWith(scaledBoundingBox.name))
  }

  def backgroundCropsGenerated(scaledBoundingBox: BoundingBox): Boolean = {
    val cropsFolderName = s"${Configuration.HomeFolder}/${Configuration.BackgroundCropImagesFolder}"
    val cropsDirectoryFiles = new File(cropsFolderName).listFiles()
    cropsDirectoryFiles.foldLeft(false)((exists, file) => exists || file.getName.startsWith(scaledBoundingBox.name))
  }

}
