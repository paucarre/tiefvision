/**
 * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
 */

var ImageSimilarityEditor = function (positive, negative) {
  var $scope = this;

  $scope.imageResults = $('.img-result');
  $scope.reference = document.getElementById('reference');
  $scope.positive = positive;
  $scope.negative = negative;
  $scope.saveButton = document.getElementById('save_button');

  // add event listeners
  $.each($scope.imageResults, function(index, item) {
    item.addEventListener("click", function(){
      if($scope.positive) {
        if($scope.negative) {
          //reset
          $scope.positive = item;
          $scope.negative =  null;
        } else {
          $scope.negative =  item;
        }
      } else {
        $scope.positive = item;
      }
      $scope.paint()
    });
  });

  $scope.saveButton.addEventListener("click", function(){
    if($scope.positive && $scope.reference && $scope.negative) {
     window.location = "/save_similarity?" +
      "reference=" + $scope.reference.name +
      "&positive=" + $scope.positive.id  +
      "&negative=" + $scope.negative.id;
    }
  });

  $scope.paint()
}

ImageSimilarityEditor.prototype.paint = function () {
  var $scope = this;
  $.each($scope.imageResults, function(index, item) {
    $(item).removeClass('img-result-positive');
    $(item).removeClass('img-result-negative');
  });
  if($scope.positive){
    $($scope.positive).addClass('img-result-positive');
  }
  if($scope.negative){
    $($scope.negative).addClass('img-result-negative');
  }
};
