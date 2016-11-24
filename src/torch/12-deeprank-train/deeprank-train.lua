-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'nn'
require 'inn'
require 'image'

local inputSize = 11 * 11 * 384
local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'

function getSimilarityModel()

  local leftImage = nn.Sequential()
  --leftImage:add(nn.SpatialConvolutionMM(11, 11, 1, 1, 1, 1, 0, 0):cuda())

  local convLayer = nn.SpatialConvolutionMM(384, 384, 1, 1, 1, 1, 0, 0):cuda()
  convLayer.weight = torch.eye(384):cuda()
  convLayer.bias = torch.zeros(384):cuda()
  leftImage:add(convLayer:cuda())
  --leftImage:add(nn.ReLU(true):cuda())
  --leftImage:add(nn.Dropout(0.5):cuda())
  --leftImage:add(nn.SpatialConvolutionMM(512, 126, 1, 1, 1, 1, 0, 0):cuda())
  --leftImage:add(nn.ReLU(true):cuda())
  --leftImage:add(nn.Dropout(0.5):cuda())
  leftImage:add(nn.Reshape(384 * 11 * 11):cuda())

  local rightImage = leftImage:clone('weight', 'bias')

  local imagesParallel = nn.ParallelTable()
  imagesParallel:add(leftImage)
  imagesParallel:add(rightImage)

  local imagesDistSimilar = nn.Sequential()
  imagesDistSimilar:add(imagesParallel)
  imagesDistSimilar:add(nn.DotProduct():cuda())

  local imagesDistDifferent = imagesDistSimilar:clone('weight', 'bias')

  local similarityModel = nn.Sequential()
  local distanceParallel = nn.ParallelTable()
  distanceParallel:add(imagesDistSimilar)
  distanceParallel:add(imagesDistDifferent)
  similarityModel:add(distanceParallel)

  return similarityModel:cuda()
end

function getCriterion()
  local criterion = nn.MarginRankingCriterion(0.1):cuda()
  return criterion
end

function gradUpdate(similarityModel, x, criterion, learningRate)
  local pred = similarityModel:forward(x)
  local err = criterion:forward(pred, torch.ones(32):cuda())
  local gradCriterion = criterion:backward(pred, torch.ones(32):cuda())
  similarityModel:zeroGradParameters()
  similarityModel:backward(x, gradCriterion)
  similarityModel:updateParameters(learningRate)
  return err
end

function getTestSet()
  return getDataSet('similarity-db-test')
end

function getTrainingSet()
  return getDataSet('similarity-db-train')
end

function getDataSet(file)
  local lines = tiefvision_commons.getLines(tiefvision_commons.resourcePath('dresses-db', file))
  local trainingSet = {}
  for i = 1, #lines do
    local reference, similar, different = string.match(lines[i], "(.+),(.+),(.+)")
    trainingSet[i] = { reference, similar, different }
  end
  return trainingSet
end

local criterion = getCriterion()
local learningRate = 0.0001

function trainBatchGradUpdate(similarityModel, batchSize, initialTrainingIndex, trainingSet, batchSet)
  for batchIndex = 1, batchSize do
    local trainingIndex = initialTrainingIndex + batchIndex - 1
    -- print(trainingSet[trainingIndex][1] .. ' ' .. trainingSet[trainingIndex][2] .. ' ' .. trainingSet[trainingIndex][3])
    local reference, similar, different = getReferenceSimilarDifferent(trainingSet[trainingIndex])
    batchSet[1][1][batchIndex] = reference
    batchSet[1][2][batchIndex] = similar
    batchSet[2][1][batchIndex] = reference
    batchSet[2][2][batchIndex] = different
  end
  local trainError = gradUpdate(similarityModel, batchSet, criterion, learningRate)
  return trainError
end

function getHeightWindow(input, heightStart)
  local windowInput = torch.Tensor(11, 11, 384):cuda()
  local trInput = input:transpose(1, 3)
  for w = 1, 11 do
    for h = 1, 11 do
      windowInput[w][h] = trInput[w][h + heightStart - 1]
    end
  end
  return windowInput:transpose(1, 3)
end

function getReferenceSimilarDifferentRaw(datasource)
  local encodedFolder = tiefvision_commons.dataPath('db/similarity/img-enc-cnn-encoder')
  local flippedEncodedFolder = tiefvision_commons.dataPath('db/similarity/img-enc-cnn-encoder-flipped')
  local reference = torch.load(encodedFolder .. '/' .. datasource[1]):cuda()
  local similar
  if (datasource[1] == datasource[2]) then
    similar = torch.load(flippedEncodedFolder .. '/' .. datasource[2]):cuda()
  else
    similar = torch.load(encodedFolder .. '/' .. datasource[2])
  end
  local different = torch.load(encodedFolder .. '/' .. datasource[3])
  return reference:transpose(1, 3):cuda(), similar:transpose(1, 3):cuda(), different:transpose(1, 3):cuda()
end

function getReferenceSimilarDifferent(datasource)
  local reference, similar, different = getReferenceSimilarDifferentRaw(datasource)

  local minHeight = math.min(math.min(reference:size()[2], similar:size()[2]), different:size()[2]) - 11 + 1
  local heightStart = math.random(minHeight)
  local referenceW = getHeightWindow(reference, heightStart)
  local similarW = getHeightWindow(similar, heightStart)
  local differentW = getHeightWindow(different, heightStart)

  return referenceW, similarW, differentW
end


function train(similarityModel)
  math.randomseed(os.time())
  local batchSize = 32
  local trainingSet = getTrainingSet()
  local trainIndex = 1
  local batchSet = torch.Tensor(2, 2, batchSize, 384, 11, 11):cuda()
  while trainIndex <= #trainingSet - batchSize do
    similarityModel:training()
    local trainError = trainBatchGradUpdate(similarityModel, batchSize, trainIndex, trainingSet, batchSet)
    print('TRAIN_ERROR:' .. trainError)
    if (((trainIndex - 1) / batchSize) % 10 == 0) then
      print("Saving model...")
      saveModel(similarityModel)
      print("Model saved")
      testModel(similarityModel)
    end
    trainIndex = trainIndex + batchSize
  end
end

function testModel(similarityModel)
  local testOk, testNum = test(similarityModel)
  print('Test Rate:' .. testOk / testNum, testOk .. ' out of ' .. testNum)
end

function test(similarityModel)
  local imagesDist = similarityModel.modules[1].modules[1]
  -- local linear1 = imagesDistSimilar.modules[1].modules[1].modules[5]
  -- local linear2 = imagesDistDifferent.modules[1].modules[2].modules[5]
  -- local weightDiff = linear1.weight - linear2.weight
  -- print(torch.mean(weightDiff), torch.std(weightDiff))

  similarityModel:evaluate()
  local testSet = getTestSet()
  local batchSetSimilar = torch.Tensor(2, 384, 11, 11):cuda()
  local batchSetDifferent = torch.Tensor(2, 384, 11, 11):cuda()
  local correctRank = 0
  local successfulTestAttempts = 0
  for testIndex = 1, #testSet do
    local similarOutput = 0.0
    local differentOutput = 0.0
    local currentCorrectRank = 0.0
    --if(testSet[testIndex][1] ~= testSet[testIndex][2]) then
    local reference, similar, different = getReferenceSimilarDifferentRaw(testSet[testIndex])
    local minHeight = math.min(math.min(reference:size()[2], similar:size()[2]), different:size()[2]) - 11
    for h = 1, 3 do
      local referenceW = getHeightWindow(reference, h)
      local similarW = getHeightWindow(similar, h)
      local differentW = getHeightWindow(different, h)

      batchSetSimilar[1] = referenceW
      batchSetSimilar[2] = similarW
      batchSetDifferent[1] = referenceW
      batchSetDifferent[2] = differentW

      local currSimilarOutput = imagesDist:forward(batchSetSimilar)[1]
      similarOutput = similarOutput + currSimilarOutput
      local currDifferentOutput = imagesDist:forward(batchSetDifferent)[1]
      differentOutput = differentOutput + currDifferentOutput
      if (currSimilarOutput > currDifferentOutput) then
        currentCorrectRank = currentCorrectRank + 1
      else
        currentCorrectRank = currentCorrectRank - 1
      end
    end
    --if(currentCorrectRank > 0) then
    if (similarOutput > differentOutput) then
      correctRank = correctRank + 1
    end
    successfulTestAttempts = successfulTestAttempts + 1
    --end
  end
  return correctRank, successfulTestAttempts
end

function saveModel(model)
  torch.save(tiefvision_commons.modelPath('similarity.model'), model)
end

function loadModel()
  local modelPath = tiefvision_commons.modelPath('similarity.model')
  if (tiefvision_commons.fileExists(modelPath)) then
    return torch.load(modelPath)
  else
    return getSimilarityModel()
  end
end

function getOptions()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Deep Rank Hinge Loss Training')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-reset', false, 'Reset the saved model (if any) and use a new model.')
  cmd:option('-epochs', 20, 'Epochs during training (iterations over all the batches in the training set).')
  cmd:text()
  return cmd:parse(arg)
end

function getModel(reset)
  if (reset) then
    return getSimilarityModel()
  else
    return loadModel()
  end
end

local options = getOptions()
local similarityModel = getModel(options.reset)
for epoch = 1, options.epochs do
  print("Epoch: " .. epoch)
  train(similarityModel)
end

print("Similarity Model Successfully Trained")
