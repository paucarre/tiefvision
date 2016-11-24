-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'inn'
require 'torch'
require 'image'

local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'

local similarity_db_lib = {}

function similarity_db_lib.getEncoder()
  return torch.load(tiefvision_commons.modelPath('encoder.model'))
end

function similarity_db_lib.encodeImage(imagePath, encoder)
  local input = image.load(imagePath)
  -- scale
  local inputWidth, inputHeight = input:size()[3], input:size()[2]
  local inputScaled = image.scale(input, 224, inputHeight * 224.0 / inputWidth)
  -- encode
  local encoderInput = tiefvision_commons.loadImage(inputScaled)
  local encoderOutput = encoder:forward(encoderInput)[2]
  encoderOutput = encoderOutput:transpose(1, 3):clone() -- make it contiguous by cloning
  for w = 1, encoderOutput:size()[1] do
    for h = 1, encoderOutput:size()[2] do
      encoderOutput[w][h] = encoderOutput[w][h] / torch.norm(encoderOutput[w][h])
    end
  end
  return encoderOutput
end

return similarity_db_lib
