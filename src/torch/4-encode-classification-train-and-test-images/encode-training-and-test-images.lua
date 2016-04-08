-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua'
require 'nn'
require 'inn'
require 'image'
local tiefvision_commons = require 'tiefvision_commons'

local batch_size = 64

-- TODO: we are wasting at most 63 samples due to batching
function loadData(encoder, lines, cl)
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
  for cl = 1, 2 do
    local lines = tiefvision_commons.getLines("../../../resources/classification-images/crops/" .. cl .. "-" .. prefix .. ".txt")
    print("../../../resources/classification-images/crops/" .. cl .. "-" .. prefix .. ".txt :" .. #lines)
    trainingFiles[cl] = lines
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
  for i=1,data:size()[1] do
   assert(torch.mean(data[i]) > 0.01, 'the mean of the data should no be zero')
  end
end

function getFile(type, cl, i) 
  return '../data/classification/' .. cl.. '/' ..  type .. '/' .. i  .. '.data' 
end 

function encodeData(type, encoder)
  for cl = 1, 2 do
    local files = getFilesAsTable(type)
    print('Class ' .. cl .. ' with files ' .. #files[cl])
    local input = loadData(encoder, files[cl], cl)
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

local encoder = torch.load('../models/encoder.model')

encodeData('train', encoder)
encodeData('test', encoder)

print('Data encoding finished')
