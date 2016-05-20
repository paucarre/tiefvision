/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
  */

package core

case class Accuracy(val count: Map[AccuracyType, Int] = Map.empty,
                    val sum: Map[AccuracyType, Int] = Map.empty) {

  def update(isCorrect: Boolean, isTest: Boolean, isFlipped: Boolean): Accuracy = {
    val performanceType = AccuracyType(isTest, isFlipped)
    val currentCount = count.get(performanceType).getOrElse(0) + 1
    val currentSum = sum.get(performanceType).getOrElse(0) + (if (isCorrect) 1 else 0)
    Accuracy(
      count = count + (performanceType -> currentCount),
      sum = sum + (performanceType -> currentSum)
    )
  }

//  def apply(performanceType: AccuracyType): Option[Double] = {
//    count.get(performanceType).map { currentCount =>
//      if(currentCount == 0) None[Double]
//      else sum.get(performanceType).getOrElse(0).toDouble / currentCount.toDouble
//    }
//  }

}

case class AccuracyType(val isTest: Boolean, isFlipped: Boolean)
