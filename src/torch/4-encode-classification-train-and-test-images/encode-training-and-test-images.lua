-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua'
require 'nn'
require 'inn'
require 'image'
tiefvision_commons = require 'tiefvision_commons'

function loadData(encoder, lines, cl)
    local inputs = torch.Tensor(#lines, 384, 11, 11):cuda()
    local outputs = torch.Tensor(#lines, 16):cuda()
    for li = 1, #lines do
        local trainFileLine = lines[li]
        local encodedInput, target = encodedInputOutput(trainFileLine, cl, encoder)
        inputs[li] = inputs[li]:set(encodedInput)
	outputs[li] = outputs[li]:set(target)
    end
    return inputs, outputs
end

function getFilesAsTable(prefix)
  local trainingFiles = {}
  for cl = 0, 13 do
      local lines = getLines("../../../resources/classification-images/crops/" .. cl .. "-" .. prefix .. ".txt")
      print("../../../resources/classification-images/crops/" .. cl .. "-" .. prefix .. ".txt :" .. #lines)
      trainingFiles[cl + 1] = lines
  end
  return trainingFiles
end

function getLines(filename)
    local trainFile = io.open(filename)
    local lines = {}
    if trainFile then
      local index = 1
      for trainFileLine in trainFile:lines() do
        if(file_exists(trainFileLine)) then
          lines[index] = trainFileLine
          index = index + 1
        end
      end
    end
    return lines
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function encodedInputOutput(name, cl, encoder)
      print(name)
      local input = tiefvision_commons.load(name)
      local encodedInput = encoder:forward(input)[2]
      local target = torch.CudaTensor(16)
      for i=1, target:size()[1] do
       if (i - 1) == cl then
        target[i] = 1
       else
        target[i] = 0
       end
      end
      return encodedInput, target
end

function testSavedData(data)
  for i=1,data:size()[1] do
   assert(torch.mean(data[i]) > 0.01, 'the mean of the data should no be zero')
  end
end

function encodeData(type, encoder)
  for cl = 2, 14, 12 do
    local files = getFilesAsTable(type)
    print('Class ' .. cl .. ' with files ' .. #files[cl])
    local input, target = loadData(encoder, files[cl], cl)
    input = input:double()
    local target = target:double()
    torch.save('../data/' .. cl  .. '-classification-' .. type .. '-in.data', input)
    torch.save('../data/' .. cl  .. '-classification-' .. type .. '-target.data', target)
    local inputLoaded = torch.load('../data/' .. cl  .. '-classification-' .. type .. '-in.data')
    local targetLoaded =  torch.load('../data/' .. cl  .. '-classification-' .. type .. '-target.data')
    testSavedData(inputLoaded)
    testSavedData(targetLoaded)
    assert(torch.eq(input, inputLoaded), 'test input not properly saved')
    assert(torch.eq(target, targetLoaded), 'test target not properly saved')
  end
end

local encoder = torch.load('../models/encoder.model')

encodeData('train', encoder)
encodeData('test', encoder)

print('Data encoding finished')
