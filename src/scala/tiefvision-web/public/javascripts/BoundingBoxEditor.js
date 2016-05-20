/**
 * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
 */
var BoundingBoxEditor = function (imageName, left, right, top, bottom) {
  var $scope = this;
  $scope.canvas = document.getElementById("bbox_canvas");
  $scope.repaintButton = document.getElementById("repaint_button");
  $scope.saveButton = document.getElementById("save_button");
  $scope.ctx = $scope.canvas.getContext("2d");
  $scope.scale = 0.5;
  $scope.boundingBox = { name: imageName }
  $scope.boundingBox.left = left;
  $scope.boundingBox.right = right;
  $scope.boundingBox.top = top;
  $scope.boundingBox.bottom = bottom;
  $scope.state = "bottomRight";
  console.log($scope.img)
  $scope.ctx.scale($scope.scale, this.scale);
  $scope.img = new Image();
  $scope.img.addEventListener('load', function() {
    $scope.width  = $scope.img.naturalWidth;
    $scope.height = $scope.img.naturalHeight;
    $scope.ctx.drawImage($scope.img, 0, 0, $scope.width, $scope.height);
    $scope.drawSavedBox()
  }, false);
  $scope.img.src = 'dresses_db/' + imageName;

  $scope.repaintButton.addEventListener("click", function(){
    window.location = "/";
  });

  $scope.saveButton.addEventListener("click", function(){
     window.location = "/save_bounding_box?" +
      "name=" + $scope.boundingBox.name +
      "&left=" + $scope.boundingBox.left +
      "&right=" + $scope.boundingBox.right +
      "&top=" + $scope.boundingBox.top +
      "&bottom=" + $scope.boundingBox.bottom +
      "&width=" + $scope.width +
      "&height=" + $scope.height;
  });

  $scope.canvas.addEventListener('mousemove',
    function(evt) {
      var mousePos = $scope.getMousePos($scope.canvas, evt);
      if($scope.state === "bottomRight") {
        $scope.drawBBox(1, 1, mousePos.x, mousePos.y);
      } else if($scope.state === "topLeft") {
        $scope.drawBBox(
          mousePos.x,
          mousePos.y,
          ($scope.boundingBox.right * $scope.scale ) - mousePos.x,
          ($scope.boundingBox.bottom * $scope.scale ) - mousePos.y
        );
      }
    }
  , false);

  $scope.canvas.addEventListener('mouseup',
    function(evt) {
      var mousePos = $scope.getMousePos($scope.canvas, evt);
      if($scope.state === "bottomRight") {
        $scope.boundingBox.right = mousePos.x / $scope.scale;
        $scope.boundingBox.bottom = mousePos.y / $scope.scale;
        $scope.state = "topLeft";
      } else if($scope.state === "topLeft") {
        $scope.boundingBox.left = mousePos.x / $scope.scale;
        $scope.boundingBox.top = mousePos.y / $scope.scale;
        $scope.state = "finish";
      }
    }
  , false);

};

BoundingBoxEditor.prototype.getMousePos = function(canvas, evt) {
  var $scope = this;
  var rect = $scope.canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top
  };
};

BoundingBoxEditor.prototype.drawBBox = function (xi, yi, xf, yf) {
  var $scope = this;
  $scope.ctx.drawImage($scope.img, 0, 0, $scope.width, $scope.height);
  $scope.ctx.beginPath();
  $scope.ctx.lineWidth = "1";
  $scope.ctx.strokeStyle = "blue";
  $scope.ctx.rect(xi / $scope.scale, yi / $scope.scale, xf / $scope.scale, yf / $scope.scale);
  $scope.ctx.stroke();
  $scope.drawSavedBox()
};

BoundingBoxEditor.prototype.drawSavedBox = function () {
  var $scope = this;
  if($scope.boundingBox.top && $scope.boundingBox.left && $scope.boundingBox.right && $scope.boundingBox.bottom) {
    $scope.ctx.beginPath();
    $scope.ctx.lineWidth = "1";
    $scope.ctx.strokeStyle = "red";
    $scope.ctx.rect(
      $scope.boundingBox.left,
      $scope.boundingBox.top,
      $scope.boundingBox.right - ($scope.boundingBox.left),
      $scope.boundingBox.bottom - ($scope.boundingBox.top)
    );
    $scope.ctx.stroke();
  }
}