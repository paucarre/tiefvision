-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;../5-train-classification/?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
require 'image'
tiefvision_commons = require 'tiefvision_commons'

function generateFlippedImages()
  local folder = "../../../resources/dresses-db/bboxes/1"
  local destFolder = "../../../resources/dresses-db/bboxes-flipped/1"
  local files = tiefvision_commons.getFiles(folder)
  for fileIndex = 1, #files do
    local file = files[fileIndex]
    print(file)
    local input = image.load(folder .. '/' .. file)
    local flippedInput = image.hflip(input)
    image.save(destFolder .. '/' .. file, flippedInput)
    collectgarbage()
  end
end


function createDb()
  local folder = "../../../resources/dresses-db/bboxes-flipped/1"
  local destFolder = "../data/db/similarity/img-enc-cnn-encoder-flipped"
  local files = tiefvision_commons.getFiles(folder)
  local encoder = torch.load('../models/encoder.model')
  local classifier = torch.load('../models/fashion-classifier-conv.model')
  for fileIndex = 1, #files do
    local file = files[fileIndex]
    print(file)
    -- load image
    local input = image.load(folder .. '/' .. file)
    -- scale
    local inputWidth, inputHeight = input:size()[3], input:size()[2]
    local inputScaled = image.scale(input, 224, inputHeight * 224.0 / inputWidth )
    -- encode
    local encoderInput = tiefvision_commons.loadImage(inputScaled)
    local encoderOutput = encoder:forward(encoderInput)[2]
    encoderOutput = encoderOutput:transpose(1, 3):clone() -- make it contiguous by cloning
    for w = 1, encoderOutput:size()[1] do
      for h = 1, encoderOutput:size()[2] do
         encoderOutput[w][h] = encoderOutput[w][h] / torch.norm(encoderOutput[w][h])
      end
    end
    torch.save(destFolder .. '/' .. file, encoderOutput)
    collectgarbage()
  end
end

-- generateFlippedImages()
createDb()
