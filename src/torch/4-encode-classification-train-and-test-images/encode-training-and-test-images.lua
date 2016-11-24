-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'nn'
require 'inn'
require 'image'
local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'

local batch_size = 64

-- TODO: we are wasting at most 63 samples due to batching
function loadData(encoder, lines)
  local batches = #lines / batch_size
  local inputs = torch.Tensor(batches, batch_size, 384, 11, 11):cuda()
  for batch = 1, batches do
    for batch_el = 1, batch_size do
      local lineIndex = ((batch - 1) * batch_size) + batch_el
      local fileName = lines[lineIndex]
      local encodedInput = encodedInput(fileName, encoder)
      inputs[batch][batch_el] = inputs[batch][batch_el]:set(encodedInput)
    end
  end
  return inputs
end

function getFilesAsTable(prefix)
  local trainingFiles = {}
  for cl = 0, 1 do
    local file_path = tiefvision_commons.resourcePath('bounding-boxes', cl .. "-" .. prefix .. ".txt")
    local lines = tiefvision_commons.getLines(file_path)
    print(file_path .. ': ' .. #lines)
    trainingFiles[cl + 1] = lines
  end
  return trainingFiles
end

function encodedInput(name, encoder)
  print(name)
  local input = tiefvision_commons.load(name)
  local encodedInput = encoder:forward(input)[2]
  collectgarbage()
  return encodedInput
end

function testSavedData(data)
  for i = 1, data:size()[1] do
    assert(torch.mean(data[i]) > 0.01, 'the mean of the data should no be zero')
  end
end

function getFile(type, cl, i)
  return tiefvision_commons.dataPath('classification', cl, type, i .. '.data')
end

function encodeData(type, encoder)
  for cl = 0, 1 do
    local files = getFilesAsTable(type)
    print('Class ' .. cl .. ' with files ' .. #files[cl + 1])
    local input = loadData(encoder, files[cl + 1])
    input = input:double()
    for i = 1, input:size()[1] do
      print("Saving batch " .. i .. " for " .. type)
      torch.save(getFile(type, cl, i), input[i]:clone()) -- use clone as otherwise it saves the whole tensor
      local inputLoaded = torch.load(getFile(type, cl, i))
      testSavedData(inputLoaded)
      assert(torch.eq(input[i], inputLoaded), 'test input not properly saved')
      collectgarbage()
    end
  end
end

local encoder = torch.load(tiefvision_commons.modelPath('encoder.model'))

encodeData('train', encoder)
encodeData('test', encoder)

print('Data encoding finished')
