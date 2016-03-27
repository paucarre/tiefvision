-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local image = require 'image'
tiefvision_commons = require 'tiefvision_commons'
bboxlib = require 'bboxlib'

function getMostSimilarImage(input, refineSteps)

  local bboxes = bboxlib.getImageBoundingBoxesTable(input, refineSteps)
  local bbox = bboxes[#bboxes]
  local xmin = bbox[1]
  local ymin = bbox[2]
  local xmax = bbox[3]
  local ymax = bbox[4]

  local inputCropped = image.crop(input, xmin, ymin, xmax, ymax)
  local croppedScaledImageOrig = image.scale(inputCropped, inputCropped:size()[2] * 224 / inputCropped:size()[3] )

  local transNormEncodedImageOrig = encodeAndNormalizeImage(croppedScaledImageOrig)

  -- iterate through all the images and take the minimum
  local optFileIndex  = 0
  local optDiff
  local files = getFiles()
  for i = 1, #files do
    local file = files[i]
    local folder = "../data/db"
    local fileNormEncodedDbImg = torch.load(folder .. '/low/' .. file)
    local aspectRatioDb  = fileNormEncodedDbImg:size()[1]  / fileNormEncodedDbImg:size()[2]
    local aspectRatioImg = (xmax - xmin) / (ymax - ymin)

    local aspectRatioPerc = math.min(aspectRatioImg, aspectRatioDb) / math.max(aspectRatioImg, aspectRatioDb)
    if(aspectRatioPerc > 0.4) then
      local diff = getOptDiffAndFileIndex(fileNormEncodedDbImg, transNormEncodedImageOrig, file)
      if not optDiff or diff < optDiff then
         optDiff = diff
         optFileIndex = i
      end
    end
  end
  local similarDres
  if(optFileIndex > 0) then
    similarDress = bboxlib.loadImageFromFile('../../../resources/dresses-db/master/1/' .. files[optFileIndex])
  end
  return similarDress, files[optFileIndex], boundingBoxes
end

function getOptDiffAndFileIndex(fileNormEncodedDbImg, transNormEncodedImage, file)
  local optDiff
  local minx = math.min(fileNormEncodedDbImg:size()[1], transNormEncodedImage:size()[1])
  local miny = math.min(fileNormEncodedDbImg:size()[2], transNormEncodedImage:size()[2])
  local minyImage
  local maxyImage
  if(fileNormEncodedDbImg:size()[2] < transNormEncodedImage:size()[2]) then
    minyImage = fileNormEncodedDbImg
    maxyImage = transNormEncodedImage
  else
    maxyImage = fileNormEncodedDbImg
    minyImage = transNormEncodedImage
  end
  local yi_max =  math.floor((maxyImage:size()[2] - minyImage:size()[2]))
  local sumDiff = 0.0
  for yi = 1, yi_max do
    for y = 1, minyImage:size()[2] do
      for x = 1, minx do
        local minyComp = minyImage[x][y]
        local maxyComp = maxyImage[x][y + yi]
        local diff = torch.norm(minyComp - maxyComp)
        sumDiff = sumDiff + diff
      end
    end
  end
  sumDiff = sumDiff / (minx * minyImage:size()[2] * yi_max )
  if not optDiff or sumDiff < optDiff then
      optDiff = sumDiff
  end
  return optDiff
end

function similarImage(sourceImage, refineSteps)
  local input = bboxlib.loadImageFromFile(sourceImage)
  local similarDress, optFile, boundingBoxes = getMostSimilarImage(input, refineSteps)
  return similarDress, optFile, boundingBoxes
end

function similarImageTest()
  local testFiles = getFiles()
  local folder = '../../../resources/regression-gown-bounding-boxes/images/original-scale/'
  local correct = 0
  for i = 1, #testFiles do
    print(testFiles[i])
    local similarDress, optFile, boundingBoxes  = similarImage(folder .. testFiles[i], 1)
    print(boundingBoxes)
    if(optFile == testFiles[i]) then
      correct = correct + 1
    else
      print(optFile)
      print(i .. ': ' .. correct .. ' / ' .. correct / i)
    end
  end
  print('Percentage correct classification: ' .. correct / #testFiles )
end

similarImageTest()
