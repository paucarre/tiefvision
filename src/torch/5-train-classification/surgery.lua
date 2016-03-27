-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
classifier = require 'classifier-conv'

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
     local inputsBatch = torch.Tensor(batchSize, 384, 6, 6):cuda()
     local targetsBatch = torch.Tensor(batchSize):cuda()
     for b = 1,trainBatches do
       for batchIndex = 1,batchSize do
          -- select random class
          local cl = math.random(13)
          -- select random sample
          local sampleIndex = math.random(trainIn[cl]:size()[1])
          -- print('Sample Index: ' .. sampleIndex)
          inputsBatch[batchIndex] =  trainIn[cl][sampleIndex]:cuda()
          targetsBatch[batchIndex] =  cl
       end
       -- print("==> online epoch # " .. epoch .. ' batch # ' .. b .. ']')
       trainBatch(model, criterion, inputsBatch, targetsBatch)
       print("TRAIN_LOSS:" .. trainingLoss)
     end
     local meanClass = getTestError(model, criterion, testIn)
     print("TEST:" .. meanClass)
     saveModel(model)
     time = sys.clock() - time
     print("Time to learn full batch = " .. (time / (60 * 60)) .. " hours\n")
   end
end

function data()
  local trainIn = {}
  local testIn = {}
  local trainTarget = {}
  local testTarget = {}
  for cl = 1, 13 do
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
   for cl = 1, #testIn do
     for i = 1, testIn[cl]:size()[1] do
   	local output = model:forward(testIn[cl][i]:cuda())
        print(output)
        local firstIndex = maxIndex(output)
        if( firstIndex == cl) then
	   classified = classified + 1
	end
        elements = elements + 1
     end
   end
   classified = classified / elements
   return classified
end

function maxIndex(output)
  local index = 1
  for i=2, output:size()[1] do
    if(output[i] > output[index]) then
       index = i
    end
  end
  return index
end

function saveModel(model)
   local filename = '../models/fashion-classifier.model'
   print('==> saving model to ' ..  filename)
   torch.save(filename, model)
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
           -- print(outputs)
           -- print(outputsBatch)
           local f = criterion:forward(outputs, targetsBatch)
           local df_do = criterion:backward(outputs, targetsBatch)
           model:backward(inputsBatch, df_do)
           -- normalize gradients and E(X)
           -- gradParameters:div(inputsBatch:size()[1])
           trainingLoss = f
           -- return E and dE/dW
           return f, gradParameters
    end

    -- optimize on current mini-batch
    optim.sgd(feval, parameters, optimState)
end

trainingLoss = 0.0
batchSize = 64
optimState = {
      learningRate = 1e-4,
      weightDecay = 0.001,
      momentum = 0.1,
      learningRateDecay = 1e-7
}

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
end

transformToConv()

local model = loadSavedModelConv()
local trainIn, trainTarget, testIn, testTarget = data()
local meanClass = getTestError(model, criterion, testIn)
print("TEST:" .. meanClass)



-- local criterion = loadCriterion()
-- train(model, criterion, 5000)
