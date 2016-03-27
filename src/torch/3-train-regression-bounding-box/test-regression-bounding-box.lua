-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
locator = require 'locatorconv'

function getTestError(model, criterion, index)
   local testIn = torch.load("../data/bbox-test-in/1.data")
   local testOut = torch.load("../data/bbox-test-out/1.data")
   local mean = torch.load("../models/bbox-train-mean")
   local std = torch.load("../models/bbox-train-std")
   local errRandVec = 0.0
   local countRand = 0
   local errVec = 0.0
   local count = 0
   for i = 1, testIn:size()[1] do
     local output = (model:forward(testIn[i])[1][1][1] * std[index]) + mean[index]
     local target = (testOut[index][i] * std[index]) +  mean[index]
     if coordinateInRange(target) then
       errVec = errVec + math.abs(target - output)
       count = count + 1
     end
     local outputRand = model:forward(testIn[((i + 2) % testIn:size()[1]) + 1])[1][1][1] * std[index]
     if coordinateInRange(outputRand) then
       errRandVec = errRandVec + math.abs((outputRand - (testOut[index][i] * std[index])))
       countRand = countRand + 1
     end
   end
   print(count)
   errVec = errVec / count
   errRandVec = errRandVec / countRand
   return errVec, errRandVec
end

function coordinateInRange(coord)
  -- return coord > -224 and coord < 224
  return true
end

function loadCriterion()
   local criterion = nn.MSECriterion()
   criterion.sizeAverage = true
   return criterion:cuda()
end

function loadSavedModelConv(index)
   return torch.load('../models/locatorconv-' .. index .. '.model')
end

function transformToConv()
  local modelConv = locator.loadModel()
  local modelSaved = loadSavedModel()

  modelConv.modules[1].weight = modelSaved.modules[2].weight
  modelConv.modules[1].bias = modelSaved.modules[2].bias

  modelConv.modules[3].weight = modelSaved.modules[4].weight
  modelConv.modules[3].bias = modelSaved.modules[4].bias

  modelConv.modules[5].weight = modelSaved.modules[6].weight
  modelConv.modules[5].bias = modelSaved.modules[6].bias

  saveModelConv(modelConv)

end

for index =1, 4 do
  local model = loadSavedModelConv(index)
  local criterion = loadCriterion()
  local error, errorRand = getTestError(model, criterion, index)
  print('Index: ' .. index)
  print('Error Actual Images: ')
  print(error)
  print('Error Random Input: ')
  print(errorRand)
  assert(error * 1.25 < errorRand, 'the error predicting images should be higher than the error from random inputs')
end

print('Test passed')
