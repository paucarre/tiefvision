-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;./?.lua'
require 'nn'
require 'inn'
require 'image'
require 'lfs'
local tiefvision_commons = require 'tiefvision_commons'

function generateDatabase(imageEncoder)
  -- unflipped
  local dataFolder = '../data/db/similarity/img-enc-cnn-encoder'
  local destFolder = '../data/db/similarity/img-similarity-deeprank'
  generateDatabaseForFolders(dataFolder, destFolder, imageEncoder)

  -- flipped
  local dataFolderFlipped = '../data/db/similarity/img-enc-cnn-encoder-flipped'
  local destFolderFlipped = '../data/db/similarity/img-flipped-similarity-deeprank'
  generateDatabaseForFolders(dataFolderFlipped, destFolderFlipped, imageEncoder)

end

function generateDatabaseForFolders(dataFolder, destFolder, imageEncoder)
  local imageFiles = tiefvision_commons.getFiles(dataFolder)
  for imageIndex = 1, #imageFiles do
    local imageFile = imageFiles[imageIndex]
    print(imageFile)
    saveEncoding(dataFolder, destFolder, imageFile, imageEncoder)
  end
end

function saveEncoding(dataFolder, destFolder, imageName, imageEncoder)
  local imageInput = torch.load(dataFolder .. '/' .. imageName)
  imageInput = imageInput:transpose(1, 3)
  local encodedImage = imageEncoder:forward(imageInput):double()
  --print((imageInput - encodedImage):sum())
  torch.save(destFolder .. '/' .. imageName, encodedImage)
end

function loadModel()
  return torch.load('../models/similarity.model')
end

--function convolutionalNetworkModel()
--  local nhiddens1 = 512
--  local nhiddens2 = 128
--  local model = nn.Sequential()
--  model:add(nn.SpatialConvolutionMM(384, nhiddens1, 11, 11, 1, 1, 0, 0))
--  model:add(nn.ReLU())
--  model:add(nn.SpatialConvolutionMM(nhiddens1, nhiddens2, 1, 1, 1, 1, 0, 0))
--  model:add(nn.ReLU())
--  return model:cuda()
--end

--function copyLayer(dest, source)
--  dest.weight = source.weight
--  dest.bias = source.bias
--end

--function convertToConvolutionNetwork(imageEncoder)
--  local model = convolutionalNetworkModel()
--  copyLayer(model.modules[1], imageEncoder.modules[2])
--  copyLayer(model.modules[3], imageEncoder.modules[5])
--  return model
--end

function removeReshapeModule(imageEncoder)
  imageEncoder:remove(2)
  return imageEncoder
end

local similarityModel = loadModel()
local imageEncoder = similarityModel.modules[1].modules[1].modules[1].modules[1]
print(imageEncoder)
imageEncoder = removeReshapeModule(imageEncoder)
generateDatabase(imageEncoder)

print("Database for Deeprank searcher generated")
