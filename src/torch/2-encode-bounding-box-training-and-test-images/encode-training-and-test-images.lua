-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

-- Uses the image encoder to encode the train and test
-- data sets for the bounding box regression

package.path = package.path .. ';../0-tiefvision-commons/?.lua'
require 'nn'
require 'inn'
require 'image'
require 'lfs'
tiefvision_commons = require 'tiefvision_commons'

function loadData(encoder, filename)
    local outputsBatch = {}
    local inputsBatch = {}
    local lines = getLines(filename)
    local batchSize = 320
    local batches =  math.ceil(#lines / batchSize)
    local linesIndex = 1
    for i = 1, batches do
      local currentBatchSize = batchSize
      if i == batches and #lines % batchSize > 0 then
        currentBatchSize = #lines % batchSize
      end
      local inputs = torch.Tensor(currentBatchSize, 384, 11, 11):cuda()
      local outputs = torch.Tensor(currentBatchSize, 4):cuda()
      for li = 1, currentBatchSize do
        local fileIndex = li + linesIndex - 1
        local trainFileLine = lines[fileIndex]
        local encodedInput, target = encodedInputOutput(trainFileLine, encoder)
        inputs[li] = inputs[li]:set(encodedInput)
	outputs[li] = outputs[li]:set(target)
      end
      linesIndex = linesIndex + currentBatchSize
      outputsBatch[i] = outputs
      inputsBatch[i] = inputs
    end
    return inputsBatch, outputsBatch
end

function getLines(filename)
    local trainFile = io.open(filename)
    local lines = {}
    if trainFile then
      local index = 1
      for trainFileLine in trainFile:lines() do
        lines[index] = trainFileLine
        index = index + 1
      end
    end
    return lines
end


function encodedInputOutput(trainFile, encoder)
      local name, widht, height, xmin, ymin, xmax, ymax = string.match(trainFile, "(.+)___(%d+)_(%d+)_(-?%d+)_(-?%d+)_(-?%d+)_(-?%d+).jpg")
      local input = tiefvision_commons.load(trainFile)
      local encodedInput = encoder:forward(input)[2]
      local target = torch.CudaTensor(4)
      target[1] = tonumber(xmin)
      target[2] = tonumber(ymin)
      target[3] = tonumber(xmax)
      target[4] = tonumber(ymax)
      return encodedInput, target
end

function testStdDevNonZero(size, folder)
  for ig = 1, size do
    local data = loadDataFromFolder(folder, ig)
    for i=1,data:size()[1] do
     assert(torch.std(data[i]) > 5.0, 'the standard deviation of the data should not be zero')
    end
  end
end

function stats(output)
   local globalMean = torch.zeros(4):cuda()
   local globalStd  = torch.zeros(4):cuda()
   for i = 1,#output do
     local mean = torch.mean(output[i], 1):cuda()
     local std = torch.std(output[i], 1):cuda()
     globalMean = torch.add(globalMean, mean[1])
     globalStd  = torch.add(globalStd, std[1])
   end
   return globalMean / #output , globalStd / #output
end

function postprocessOutput(output, mean, std)
  local globalOutZeroMeanOneStd = {}
  for i = 1, #output do
    local meanRep = torch.repeatTensor(mean, output[i]:size()[1], 1)
    local stdRep = torch.repeatTensor(std, output[i]:size()[1], 1)
    local outZeroMeanOneStd = torch.add(output[i], -meanRep)
    outZeroMeanOneStd:cdiv(stdRep)
    globalOutZeroMeanOneStd[i] = outZeroMeanOneStd
  end
  return globalOutZeroMeanOneStd
end

function getBoundingBoxes(output)
   local avg = torch.load("../models/bbox-train-mean")
   local std = torch.load("../models/bbox-train-std")
   local avgExt = torch.repeatTensor(avg, output:size()[1], 1)
   local stdExt = torch.repeatTensor(std, output:size()[1], 1)
   local outputTrans = torch.cmul(output, stdExt) + avgExt
   return outputTrans
end

function testinLoadTest(testin)
  testStdDevNonZero(#testin, 'bbox-test-in')
  for i = 1, #testin do
    local testinLoad = loadDataFromFolder('bbox-test-in', i)
    assert(torch.eq(testinLoad, testin[i]), 'test input not properly saved')
  end
end

function testEq(actual, savedFolder)
  for i = 1, #actual do
    local saved = loadDataFromFolder(savedFolder, i)
    assert(torch.eq(actual[i], saved), 'tensor not properly saved')
  end
end

function traininLoadTest(trainin)
  testStdDevNonZero(#trainin, 'bbox-train-in')
  testEq(trainin, 'bbox-train-in')
end

function testBoundingBoxes(trainout, trainoutProc)
  for i = 1, #trainout do
    local reconstructedBoundingBox = getBoundingBoxes(trainoutProc[i])
    assert(torch.mean(torch.abs(reconstructedBoundingBox -trainout[i])) < 0.0001, 'the reconstructed bounding box should be the original one')
  end
end

function saveEncodedData()
  local encoder = torch.load('../models/encoder.model')
  local trainin, trainout = loadData(encoder, '../../../resources/bounding-boxes/TRAIN.txt')
  local mean, std = stats(trainout, meanfile, stdfile)
  torch.save('../models/bbox-train-mean', mean)
  torch.save('../models/bbox-train-std', std)

  local trainoutProc = postprocessOutput(trainout, mean, std)
  local meantest, stdtest = stats(trainoutProc)
  assert(torch.mean(meantest) < 0.0001, 'mean should be zero')
  assert(math.abs(torch.mean(stdtest) - 1) < 0.0001, 'std should be one')


  local testin, testout = loadData(encoder, '../../../resources/bounding-boxes/TEST.txt')
  local testoutProc = postprocessOutput(testout, mean, std)
  local meantest, stdtest = stats(testoutProc)
  assert(torch.mean(meantest) < 0.2, 'test mean should be close to zero')
  assert(math.abs(torch.mean(stdtest) - 1) < 0.3, 'test std should be close to one')

  for i = 1, #testin do
    local testoutTr = testoutProc[i]:transpose(1, 2)
    torch.save('../data/bbox-test-in/'   .. i .. '.data', testin[i])
    torch.save('../data/bbox-test-out/'  .. i .. '.data', testoutTr)
  end
  for i = 1, #trainin do
    local trainoutTr = trainoutProc[i]:transpose(1, 2)
    torch.save('../data/bbox-train-in/'  .. i .. '.data', trainin[i])
    torch.save('../data/bbox-train-out/' .. i .. '.data', trainoutTr)
  end
  return testin, testoutProc, trainin, trainoutProc, trainout
end

function loadDataFromFolder(bboxFolder, i)
  local folder = "../data/" .. bboxFolder
  local fileData = torch.load(folder .. '/' .. i .. '.data')
  return fileData
end

local testin, testoutProc, trainin, trainoutProc, trainout = saveEncodedData()

testinLoadTest(testin)
traininLoadTest(trainin)

testEq(testoutProc, 'bbox-test-out')
testEq(trainoutProc, 'bbox-train-out')

testBoundingBoxes(trainout, trainoutProc)

print('Data encoding finished')
