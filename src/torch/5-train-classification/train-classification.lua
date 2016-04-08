-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
local tiefvision_commons = require 'tiefvision_commons'
local classifier = require 'classifier-conv'

local trainingLoss = 0.0
local batchSize = 64
local optimState = {
  learningRate = 1e-2,
  weightDecay = 0.0,
  momentum = 0.1,
  learningRateDecay = 1e-7
}

local inputsBatch = torch.Tensor(batchSize, 384, 11, 11):cuda()
local targetsBatch = torch.Tensor(batchSize):cuda()

function train(model, criterion, epochs)
   model:training()
   local trainIn = {}
   for epoch = 1, epochs do
     local time = sys.clock()
     -- load the encoded input with their targets
     -- local meanClass, loss = getTestError(model, criterion, testIn[2], testTarget[2])
     -- print("TEST:" .. loss .. ':' .. meanClass)
     print('==> doing epoch on training data:')
     print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
     math.randomseed(os.time())
     local batchesIn1 = getBatchesInClassAndType(1, 'train')
     local batchesIn2 = getBatchesInClassAndType(2, 'train')
     for iter = 1, batchesIn1 + batchesIn2  do
       print("Epoch " .. epoch  .. ". Batch Iteration: " .. iter)
       local batchIndexClass1 = math.random(batchesIn1)
       local batchIndexClass2 = math.random(batchesIn2)
       local batchClass1 = torch.load(getFilename('train', 1, batchIndexClass1)):cuda()
       local batchClass2 = torch.load(getFilename('train', 2, batchIndexClass2)):cuda()
       local batches = {batchClass1, batchClass2}
       for batchIndex = 1,batchSize do
          -- select random class
          local cl = math.random(2)
          -- select random sample in batch
          local sampleIndex = math.random(64)
          inputsBatch[batchIndex] = batches[cl][sampleIndex]
          targetsBatch[batchIndex] =  cl
       end
       -- print("==> online epoch # " .. epoch .. ' batch # ' .. b .. ']')
       trainBatch(model, criterion, inputsBatch, targetsBatch)
       print("TRAIN_LOSS:" .. trainingLoss)
       if(iter % 10 == 0) then
         local meanClass = getTestError(model, criterion)
         print("TEST:" .. meanClass)
         saveModelConv(model)
       end
     end
     local meanClass = getTestError(model, criterion, testIn)
     print("TEST:" .. meanClass)
     saveModelConv(model)
     time = sys.clock() - time
     print("Time to learn full batch = " .. (time / (60 * 60)) .. " hours\n")
     collectgarbage() 
  end
end

function getFilename(type, cl, i)
  return '../data/classification/' .. cl .. '/' ..  type .. '/' .. i  .. '.data'
end

function getBatchesInClassAndType(class, type) 
  local folder = '../data/classification/' .. class .. '/' ..  type 
  local lines = tiefvision_commons.getFiles(folder)
  return #lines
end

function getDataFromClassAndType(class, type)
  local batches = getBatchesInClassAndType(class, type)
  local tensor = torch.Tensor(batches, 64, 384, 11, 11)
  for l= 1, batches do
    local loadedTensor = torch.load(getFilename(type, class, l))
    tensor[l] = loadedTensor
    collectgarbage() 
  end
  return tensor:cuda() 
end

function getTestError(model, criterion, testIn)
   local testIn = {}
   testIn[1] = getDataFromClassAndType(1, 'test') 
   testIn[2] = getDataFromClassAndType(2, 'test') 
   local classified = 0
   local elements = 0
   for cl = 1, 2 do
     local testInCl = testIn[cl]
     for batch=1, testInCl:size()[1] do 
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
    if(maxIndex[e] == cl) then
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
    for i=2, output:size()[1] do
      if(output[i] > output[index]) then
        index = i
      end
    end
    maxIndex[e] = index
  end
  return maxIndex
end

function saveModelConv(model)
  local filename = '../models/classifier.model'
  print('==> saving model to ' ..  filename)
  torch.save(filename, model)
end

function trainBatch(model, criterion, inputsBatch, targetsBatch)
  local parameters, gradParameters = model:getParameters()
  -- create closure to evaluate E(X) and dE/dW
  local feval = function(x)
    if x ~= parameters then
      parameters:copy(x)
    end
    gradParameters:zero()
    local outputs = model:forward(inputsBatch)
    local f = criterion:forward(outputs, targetsBatch)
    local df_do = criterion:backward(outputs, targetsBatch)
    model:backward(inputsBatch, df_do)
    trainingLoss = f
    collectgarbage()
    -- return E and dE/dW
    return f, gradParameters
  end
  -- optimize on current mini-batch
  optim.sgd(feval, parameters, optimState)
end

function loadCriterion()
  local criterion = nn.CrossEntropyCriterion()
  criterion.sizeAverage = true
  return criterion:cuda()
end

function loadSavedModelConv()
  return torch.load('../models/classifier.model')
end

--local model = loadSavedModelConv()
local model = classifier.loadModel()
local meanClass = getTestError(model, criterion)
print("TEST:" .. meanClass)
local criterion = loadCriterion()
train(model, criterion, 10)
