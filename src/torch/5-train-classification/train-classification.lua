-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'inn'
require 'optim'
require 'torch'
require 'xlua'

local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'
local classifier = require '5-train-classification/classifier-conv'

local batchSize = 64

local inputsBatch = torch.Tensor(batchSize, 384, 11, 11):cuda()
local targetsBatch = torch.Tensor(batchSize):cuda()

function train(model, criterion, epochs, optimState)
  model:training()
  for epoch = 1, epochs do
    math.randomseed(os.time())
    local batchesIn1 = getBatchesInClassAndType(1, 'train')
    local batchesIn2 = getBatchesInClassAndType(2, 'train')
    for iter = 1, batchesIn1 + batchesIn2 do
      local batchIndexClass1 = math.random(batchesIn1)
      local batchIndexClass2 = math.random(batchesIn2)
      local batchClass1 = torch.load(getFilename('train', 1, batchIndexClass1)):cuda()
      local batchClass2 = torch.load(getFilename('train', 2, batchIndexClass2)):cuda()
      local batches = { batchClass1, batchClass2 }
      for batchIndex = 1, batchSize do
        -- select random class
        local cl = math.random(2)
        -- select random sample in batch
        local sampleIndex = math.random(64)
        inputsBatch[batchIndex] = batches[cl][sampleIndex]
        targetsBatch[batchIndex] = cl
      end
      local trainingLoss = trainBatch(model, criterion, inputsBatch, targetsBatch, optimState)
      if (iter % 10 == 0) then
        local meanClass = getTestError(model)
        print("Epoch: " .. epoch .. ". Batch: " .. iter .. ". Train Loss: " .. trainingLoss .. ". Test Accuracy: " .. meanClass)
        saveModelConv(model)
      else
        print("Epoch " .. epoch .. " out of " .. epochs .. ". Batch Iteration: " .. iter .. ". Train Loss: " .. trainingLoss)
      end
    collectgarbage()
    end
  end
end

function getFilename(type, cl, i)
  return tiefvision_commons.dataPath('classification', (cl - 1), type, i .. '.data')
end

function getBatchesInClassAndType(class, type)
  local folder = tiefvision_commons.dataPath('classification', (class - 1), type)
  local lines = tiefvision_commons.getFiles(folder)
  return #lines
end

function getDataFromClassAndType(class, type)
  local batches = getBatchesInClassAndType(class, type)
  local tensor = torch.Tensor(batches, 64, 384, 11, 11)
  for l = 1, batches do
    local loadedTensor = torch.load(getFilename(type, class, l))
    tensor[l] = loadedTensor
    collectgarbage()
  end
  return tensor:cuda()
end

function getTestError(model)
  local testIn = {}
  testIn[1] = getDataFromClassAndType(1, 'test')
  testIn[2] = getDataFromClassAndType(2, 'test')
  local classified = 0
  local elements = 0
  for cl = 1, 2 do
    local testInCl = testIn[cl]
    for batch = 1, testInCl:size()[1] do
      local output = model:forward(testInCl[batch])
      output = torch.squeeze(output)
      local firstIndex = maxIndex(output)
      classified = classified + correctClassNum(firstIndex, cl)
      elements = elements + testInCl[batch]:size()[1]
    end
  end
  classified = classified / elements
  return classified
end

function correctClassNum(maxIndex, cl)
  local correctClass = 0
  for e = 1, maxIndex:size()[1] do
    if (maxIndex[e] == cl) then
      correctClass = correctClass + 1
    end
  end
  return correctClass
end

function maxIndex(outputs)
  local maxIndex = torch.Tensor(outputs:size()[1])
  for e = 1, outputs:size()[1] do
    local output = outputs[e]
    local index = 1
    for i = 1, output:size()[1] do
      if (output[i] > output[index]) then
        index = i
      end
    end
    maxIndex[e] = index
  end
  return maxIndex
end

function saveModelConv(model)
  local filename = tiefvision_commons.modelPath('classifier.model')
  print('==> Saving Model: ' .. filename)
  torch.save(filename, model)
end

function trainBatch(model, criterion, inputsBatch, targetsBatch, optimState)
  local parameters, gradParameters = model:getParameters()
  local feval = function(x)
    if x ~= parameters then
      parameters:copy(x)
    end
    gradParameters:zero()
    local outputs = model:forward(inputsBatch)
    local f = criterion:forward(outputs, targetsBatch)
    local df_do = criterion:backward(outputs, targetsBatch)
    model:backward(inputsBatch, df_do)
    collectgarbage()
    return f, gradParameters
  end
  local x, fx = optim.sgd(feval, parameters, optimState)
  return fx[1]
end

function loadCriterion()
  local criterion = nn.CrossEntropyCriterion()
  criterion.sizeAverage = true
  return criterion:cuda()
end

function loadSavedModelConv()
  local modelPath = tiefvision_commons.modelPath('classifier.model')
  if(tiefvision_commons.fileExists(modelPath)) then
    return torch.load(modelPath)
  else
    return classifier.loadModel()
  end
end

function getOptions()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Foreground and Background Classification Training')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-reset', false, 'Reset the saved model (if any) and use a new model.')
  cmd:option('-epochs', 10, 'Number of epochs to train.')
  -- optim state
  cmd:option('-learningRate', 1e-2, 'Learning rate.')
  cmd:option('-weightDecay',  0.0, 'Weight Decay (L1 regularization).')
  cmd:option('-momentum', 0.1, 'Momentum.')
  cmd:option('-learningRateDecay', 1e-7, 'Learning Rate Decay.')
  cmd:text()
  return cmd:parse(arg)
end

function getModel(options)
  if (options.reset) then
    return classifier.loadModel()
  else
    return loadSavedModelConv()
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
local meanClass = getTestError(model)
print("Test Accuracy:" .. meanClass)
local criterion = loadCriterion()
train(model, criterion, options.epochs, optimState)
