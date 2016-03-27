-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua'
require 'nn'
require 'inn'
require 'image'
tiefvision_commons = require 'tiefvision_commons'
require 'lfs'

local imageSize = 64

function getFiles(folder)
  local files = {}
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file,"mode") == "file") then
      table.insert(files, file)
    end
  end
  return files
end

function encodeUnnormalizedData()
  local folder = "../../../resources/dresses-db/bboxes/1"
  local files = getFiles(folder)
  local sumLow  = zeroTensor()
  for i = 1, #files do
    print('Encode Unnormalized: ' .. files[i])
    local imagePath = folder .. '/' .. files[i]
    local input = image.load(imagePath)
    local scaledImage = image.scale(input, imageSize, imageSize * 2):cuda()
    sumLow = sumLow + scaledImage
    torch.save('../data/db/similarity/img/' .. files[i], scaledImage )
    collectgarbage()
  end
  return sumLow, #files
end

function zeroTensor()
  return torch.zeros(3, imageSize * 2, imageSize):cuda()
end

function computeAverageAndStandardDeviation(sumLow, countLow)
  local stdDev = zeroTensor()
  local average = torch.div(sumLow, countLow)
  local folder = "../../../resources/dresses-db/bboxes/1"
  local files = getFiles(folder)
  local currentQuadraticDiff = zeroTensor()
  for i = 1, #files do
    print('Compute Average and Std. Dev. : ' .. files[i])
    local encodedInput = torch.load('../data/db/similarity/img/' .. files[i])
    local currentQuadraticDiff = torch.pow(encodedInput - average, 2)
    stdDev = stdDev + currentQuadraticDiff
    collectgarbage()
  end
  stdDev = torch.sqrt(torch.div(stdDev, countLow))
  return average, stdDev
end

function saveUnnormalized(encodedInput, file, sumLow)
  sumLow = torch.add(sumLow, encodedInput)
  torch.save('../data/db/' .. file, encodedInput)
  return sumLow
end

function encodeNormalizedData(average, stdDev)
  local folder = "../../../resources/dresses-db/bboxes/1"
  local files = getFiles(folder)
  local sumLow  = zeroTensor()
  for i = 1, #files do
    print('Encode Normalized: ' ..  files[i])
    saveNormalized('similarity/img-norm/' .. files[i], 'similarity/img/' .. files[i], average, stdDev)
    collectgarbage()
  end
  return sumLow, #files
end

function testNormalization(average, stdDev)
  local folder = "../../../resources/dresses-db/bboxes/1"
  local files = getFiles(folder)
  for i = 1, #files do
    print('Test Normalization: ' ..  files[i])
    testNormalizedFile('../data/db/similarity/img-norm/' .. files[i],
		       '../data/db/similarity/img/' .. files[i], average, stdDev)
    collectgarbage()
  end
end

function testNormalizedFile(normalizedFile, rawFile, average, stdDev)
  local normalized = torch.load(normalizedFile)
  local raw = torch.load(rawFile)
  local diff = torch.cmul(normalized, stdDev) + average - raw
  local meanSquaredError = math.sqrt(torch.mean(torch.pow(diff, 2)))
  assert(meanSquaredError < 0.00001, 'file ' .. normalizedFile .. ' not properly normalized as its reconstruction error is ' ..  meanSquaredError)
end


function saveNormalized(fileDest, fileSrc, average, stdDev)
  local encodedInput = torch.load('../data/db/' .. fileSrc)
  local encodedInputCentral = encodedInput - average
  local normEncodedInput = torch.cdiv(encodedInputCentral, stdDev)
  local meanCloseToZero = math.abs(torch.mean(normEncodedInput))
  assert(meanCloseToZero < 10.0, "the mean should be close to zero")
  local stdCloseToOne = torch.std(normEncodedInput)
  assert(stdCloseToOne < 15.0, "the standard deviation should be close to one")
  torch.save('../data/db/' .. fileDest, normEncodedInput)
end

local sumLow, countLow = encodeUnnormalizedData()
local average, stdDev = computeAverageAndStandardDeviation(sumLow, countLow)
torch.save('../models/similarity-average', average)
torch.save('../models/similarity-stddev', stdDev)
local average = torch.load('../models/similarity-average')
local stdDev  = torch.load('../models/similarity-stddev')
encodeNormalizedData(average, stdDev)
testNormalization(average, stdDev)

print('DB encoding finished')
