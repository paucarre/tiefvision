-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local locator = require 'locatorconv'

local trainingLoss = 0.0
local batchSize = 32
local optimState = {
      learningRate = 1e-6,
      weightDecay = 1.0,
      momentum = 0.1,
      learningRateDecay = 1e-7
}

local inputsBatch = torch.Tensor(batchSize, 384, 11, 11):cuda()
local outputsBatch = torch.Tensor(batchSize, 1):cuda()
function train(trainIn, trainOut, model, criterion, index)
  local trainIndex = 1
  local trainBatches = math.floor(trainIn:size()[1] / batchSize)
  for b = 1,trainBatches do
    for batchIndex = 1,batchSize do
      if batchIndex + trainIndex  - 1 <= trainIn:size()[1] then
        inputsBatch[batchIndex] = trainIn[batchIndex + trainIndex - 1]
        outputsBatch[batchIndex] = trainOut[batchIndex + trainIndex - 1]
      end
    end
    trainIndex = trainIndex + batchSize
    trainBatch(model, criterion, inputsBatch, outputsBatch)
    -- print("TRAIN_LOSS:" .. trainingLoss)
    -- local testError = getTestError(model, criterion, index)
    -- print("TEST_LOSS:" .. testError)
  end
  saveModel(model, index)
end

function loadDataFromFolder(dataFolder)
  local folder = '../data/' .. dataFolder
  local fileCount = 0
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file,"mode") == "file") then
      fileCount = fileCount + 1
    end
  end
  local data = {n=fileCount}
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file,"mode") == "file") then
      local dataInFile = torch.load(folder .. '/' .. file)
      local indexStr =  string.match(file, '(%d+)%.data')
      local i = math.floor(indexStr)
      data[i] = dataInFile
    end
  end
  return data
end

function getTestError(model, criterion, index)
   local testIn = torch.load("../data/bbox-test-in/1.data")
   local testOut = torch.load("../data/bbox-test-out/1.data")
   local output = model:forward(testIn)
   local err = criterion:forward(output, testOut[index])
   return err
end

function saveModel(model, index)
   local filename = '../models/locatorconv-' .. index .. '.model'
   print('==> Saving Model in ' ..  filename)
   torch.save(filename, model)
   print('==> Model Saved in ' ..  filename)
end

function trainBatch(model, criterion, inputsBatch, outputsBatch)
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
           trainingLoss = f
           return f, gradParameters
    end

    optim.sgd(feval, parameters, optimState)
end

function loadCriterion()
   local criterion = nn.MSECriterion()
   criterion.sizeAverage = false
   return criterion:cuda()
end

function loadSavedModel(index)
   return torch.load('../models/locatorconv-' .. index .. '.model')
end

local index = arg[1]
if (not(index)) then
  print("Add a number from 1 to 4 as argument to the command to train one of the four points.")
end
print('Training Index ' .. index)
local mean = torch.load('../models/bbox-train-mean')
local model = loadSavedModel(index)
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
    train(trainIn, trainOut, model, criterion, index)
    local testError = getTestError(model, criterion, index)
    print("Test Loss = " .. testError)
  end
  time = sys.clock() - time
  print("Time to learn full batch = " .. (time / (60 * 60)) .. " hours\n")
end
