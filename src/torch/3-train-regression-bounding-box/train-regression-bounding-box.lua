-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'

local locatorconv = require '3-train-regression-bounding-box/locatorconv'

local batchSize = 32

local inputsBatch = torch.Tensor(batchSize, 384, 11, 11):cuda()
local outputsBatch = torch.Tensor(batchSize, 1):cuda()
local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'

function train(trainIn, trainOut, model, criterion, index, optimState)
  local trainIndex = 1
  local trainingLoss = 0.0
  local trainBatches = math.floor(trainIn:size()[1] / batchSize)
  for b = 1, trainBatches do
    for batchIndex = 1, batchSize do
      if batchIndex + trainIndex - 1 <= trainIn:size()[1] then
        inputsBatch[batchIndex] = trainIn[batchIndex + trainIndex - 1]
        outputsBatch[batchIndex] = trainOut[batchIndex + trainIndex - 1]
      end
    end
    trainIndex = trainIndex + batchSize
    trainingLoss = trainBatch(model, criterion, inputsBatch, outputsBatch, optimState)
    print("Batch: " .. b .. " out of " .. trainBatches .. ". Train Loss: " .. trainingLoss)
  end
  saveModel(model, index)
  return trainingLoss
end

function loadDataFromFolder(dataFolder)
  local folder = tiefvision_commons.dataPath(dataFolder)
  local fileCount = 0
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file, "mode") == "file") then
      fileCount = fileCount + 1
    end
  end
  local data = { n = fileCount }
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file, "mode") == "file") then
      local dataInFile = torch.load(folder .. '/' .. file)
      local indexStr = string.match(file, '(%d+)%.data')
      local i = math.floor(indexStr)
      data[i] = dataInFile
    end
  end
  return data
end

function getTestError(model, criterion, index)
  local testIn = torch.load(tiefvision_commons.dataPath('bbox-test-in/1.data'))
  local testOut = torch.load(tiefvision_commons.dataPath('bbox-test-out/1.data'))
  local output = model:forward(testIn)
  local err = criterion:forward(output, testOut[index])
  return err
end

function saveModel(model, index)
  local filename = tiefvision_commons.modelPath('locatorconv-' .. index .. '.model')
  print('==> Saving Model: ' .. filename)
  torch.save(filename, model)
  print('==> Model Saved: ' .. filename)
end

function trainBatch(model, criterion, inputsBatch, outputsBatch, optimState)
  local parameters, gradParameters = model:getParameters()
  local feval = function(x)
    if x ~= parameters then
      parameters:copy(x)
    end
    gradParameters:zero()
    local outputs = model:forward(inputsBatch)
    local f = criterion:forward(outputs, outputsBatch)
    local df_do = criterion:backward(outputs, outputsBatch)
    model:backward(inputsBatch, df_do)
    return f, gradParameters
  end

  local x, fx = optim.sgd(feval, parameters, optimState)
  return fx[1]
end

function loadCriterion()
  local criterion = nn.MSECriterion()
  criterion.sizeAverage = false
  return criterion:cuda()
end

function loadSavedModel(index)
  local modelPath = tiefvision_commons.modelPath('locatorconv-' .. index .. '.model')
  if(tiefvision_commons.fileExists(modelPath)) then
    return torch.load(modelPath)
  else
    return locatorconv.loadModel()
  end
end

function getOptions()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Bounding Box Regression Training')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-reset', false, 'Reset the saved model (if any) and use a new model.')
  cmd:option('-index', 1, 'Index of the bunding box point to train (1:x-min, 2: y-min, 3: x-max, 4:y-max)')
  -- optim state
  cmd:option('-learningRate', 1e-6, 'Learning rate.')
  cmd:option('-weightDecay',  1.0, 'Weight Decay (L1 regularization).')
  cmd:option('-momentum', 0.1, 'Momentum.')
  cmd:option('-learningRateDecay', 1e-7, 'Learning Rate Decay.')
  cmd:text()
  return cmd:parse(arg)
end

function getModel(options)
  if (options.reset) then
    return locatorconv.loadModel()
  else
    return loadSavedModel(options.index)
  end
end

function getIndexLabel(index)
  if (index == 1) then
    return "x-min"
  elseif (index == 2) then
    return "y-min"
  elseif (index == 3) then
    return "x-max"
  else
    return "y-max"
  end
end

function trainIndex(index, model, optimState)
  local indexLabel = getIndexLabel(index)
  print("Training " .. indexLabel)
  local criterion = loadCriterion()
  model:training()
  local epochs = 30
  for epoch = 1, epochs do
    local trainInData = loadDataFromFolder("bbox-train-in")
    local trainOutData = loadDataFromFolder("bbox-train-out")
    local time = sys.clock()
    for i = 1, #trainInData do
      local trainIn = trainInData[i]
      local trainOuts = trainOutData[i]
      local trainOut = trainOuts[index]
      print("-------------------------------------------------------------------")
      print("Training index " .. index .. " for epoch " .. epoch .. " and batch " .. i)
      local trainingLoss = train(trainIn, trainOut, model, criterion, index, optimState)
      local testError = getTestError(model, criterion, index)
      print("Test Loss: " .. testError .. ". Train Loss: " .. trainingLoss)
    end
    time = sys.clock() - time
    print("Time to learn full batch = " .. (time / (60 * 60)) .. " hours\n")
  end
end

function getOptimSatate(options)
  local optimState = {
    learningRate = options.learningRate,
    weightDecay = options.weightDecay,
    momentum = options.momentum,
    learningRateDecay = options.learningRateDecay
  }
  return optimState
end

local options = getOptions()
local optimState = getOptimSatate(options)
local model = getModel(options)
trainIndex(options.index, model, optimState)

