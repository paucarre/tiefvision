-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
classifier = require 'classifier-conv'

local trainingLoss = 0.0
local batchSize = 64
local optimState = {
  learningRate = 1e-4,
  weightDecay = 0.001,
  momentum = 0.1,
  learningRateDecay = 1e-7
}

local inputsBatch = torch.Tensor(batchSize, 384, 11, 11):cuda()
local targetsBatch = torch.Tensor(batchSize):cuda()

function train(model, criterion, epochs)
   local trainLogger = optim.Logger('train.log')
   model:training()
   local trainIn, trainTarget, testIn, testTarget = data()
   for epoch = 1, epochs do
     local time = sys.clock()
     -- load the encoded input with their targets
     -- local meanClass, loss = getTestError(model, criterion, testIn[2], testTarget[2])
     -- print("TEST:" .. loss .. ':' .. meanClass)
     print('==> doing epoch on training data:')
     print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
     math.randomseed(os.time())
     local trainIndex = 1
     local trainBatches = math.floor(trainIn[2]:size()[1] / batchSize)
     for b = 1,trainBatches do
       for batchIndex = 1,batchSize do
          -- select random class
          local cl = math.random(2)
          if(cl == 1) then
            cl = 2
          else
            cl = 14
          end
          -- select random sample
          local sampleIndex = math.random(trainIn[cl]:size()[1])
          -- print('Sample Index: ' .. sampleIndex)
          inputsBatch[batchIndex] =  trainIn[cl][sampleIndex]:cuda()
          targetsBatch[batchIndex] =  cl
       end
       -- print("==> online epoch # " .. epoch .. ' batch # ' .. b .. ']')
       trainBatch(model, criterion, inputsBatch, targetsBatch)
       -- print("TRAIN_LOSS:" .. trainingLoss)
       if(b % 10 == 0) then
         local meanClass = getTestError(model, criterion, testIn)
         print("TEST:" .. meanClass)
       end
     end
     local meanClass = getTestError(model, criterion, testIn)
     print("TEST:" .. meanClass)
     saveModelConv(model)
     time = sys.clock() - time
     print("Time to learn full batch = " .. (time / (60 * 60)) .. " hours\n")
   end
end

function data()
  local trainIn = {}
  local testIn = {}
  local trainTarget = {}
  local testTarget = {}
  for cl = 1, 14 do
    trainIn[cl]     = torch.load('../data/' .. cl  .. '-classification-train-in.data')
    testIn[cl]      = torch.load('../data/' .. cl  .. '-classification-test-in.data')
    trainTarget[cl] = torch.load('../data/' .. cl  .. '-classification-train-target.data')
    testTarget[cl]  = torch.load('../data/' .. cl  .. '-classification-test-target.data')
  end
  return trainIn, trainTarget, testIn, testTarget
end

function getTestError(model, criterion, testIn)
   local classified = 0
   local elements = 0
   for cl = 2, #testIn, 12 do
     local output = model:forward(testIn[cl]:cuda())
     output = torch.squeeze(output)
     local firstIndex = maxIndex(output)
     classified = classified + correctClassNum(firstIndex, cl)
     elements = elements + testIn[cl]:size()[1]
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
   local filename = '../models/fashion-classifier-conv.model'
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

function loadSavedModel()
   return torch.load('../models/fashion-classifier.model')
end

function loadSavedModelConv()
   return torch.load('../models/fashion-classifier-conv.model')
end

function transformToConv()
  local modelConv = classifier.loadModel()
  local modelSaved = loadSavedModel()

  modelConv.modules[1].weight = modelSaved.modules[2].weight
  modelConv.modules[1].bias = modelSaved.modules[2].bias

  modelConv.modules[3].weight = modelSaved.modules[4].weight
  modelConv.modules[3].bias = modelSaved.modules[4].bias

  modelConv.modules[5].weight = modelSaved.modules[6].weight
  modelConv.modules[5].bias = modelSaved.modules[6].bias

  saveModelConv(modelConv)

  local model = loadSavedModel()
  local trainIn, trainTarget, testIn, testTarget = data()
  local meanClass = getTestError(model, criterion, testIn)
  print("TEST:" .. meanClass)
end

local model = loadSavedModelConv()
-- local model = classifier.loadModel()
local trainIn, trainTarget, testIn, testTarget = data()
local meanClass = getTestError(model, criterion, testIn)
print("TEST:" .. meanClass)
local criterion = loadCriterion()
train(model, criterion, 10)
