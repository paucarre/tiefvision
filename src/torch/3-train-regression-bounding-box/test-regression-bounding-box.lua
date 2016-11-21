-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local libsFolder = require('paths').thisfile('..')
package.path = package.path .. ';' ..
  libsFolder .. '/0-tiefvision-commons/?.lua;' ..
  libsFolder .. '/3-train-regression-bounding-box/?.lua'

require 'inn'
require 'optim'
require 'torch'
require 'xlua'
tiefvision_commons = require 'tiefvision_commons'

function getTestError(model, index)
  local testIn = torch.load(tiefvision_commons.dataPath('bbox-test-in/1.data'))
  local testOut = torch.load(tiefvision_commons.dataPath('bbox-test-out/1.data'))
  local mean = torch.load(tiefvision_commons.modelPath('bbox-train-mean'))
  local std = torch.load(tiefvision_commons.modelPath('bbox-train-std'))
  local errRandVec = 0.0
  local countRand = 0
  local errVec = 0.0
  local count = 0
  for i = 1, testIn:size()[1] do
    local output = (model:forward(testIn[i])[1][1][1] * std[index]) + mean[index]
    local target = (testOut[index][i] * std[index]) + mean[index]
    errVec = errVec + math.abs(target - output)
    count = count + 1
    local outputRand = model:forward(testIn[((i + 2) % testIn:size()[1]) + 1])[1][1][1] * std[index]
    errRandVec = errRandVec + math.abs((outputRand - (testOut[index][i] * std[index])))
    countRand = countRand + 1
  end
  errVec = errVec / count
  errRandVec = errRandVec / countRand
  return errVec, errRandVec
end

function loadSavedModelConv(index)
  return torch.load(tiefvision_commons.modelPath('locatorconv-' .. index .. '.model'))
end

for index = 1, 4 do
  local model = loadSavedModelConv(index)
  local error, errorRand = getTestError(model, index)
  print('Index: ' .. index)
  print('  Error Actual Images: ' .. error)
  print('  Error Random Input : ' .. errorRand)
  assert(error * 1.25 < errorRand, 'the error predicting images should be higher than the error from random inputs')
end

print('Test passed')
