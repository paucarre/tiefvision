-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'nn'
require 'inn'
require 'image'
require 'lfs'

local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'

function generateDatabase(imageEncoder)
  -- unflipped
  local dataFolder = tiefvision_commons.dataPath('db/similarity/img-enc-cnn-encoder')
  local destFolder = tiefvision_commons.dataPath('db/similarity/img-similarity-deeprank')
  generateDatabaseForFolders(dataFolder, destFolder, imageEncoder)

  -- flipped
  local dataFolderFlipped = tiefvision_commons.dataPath('db/similarity/img-enc-cnn-encoder-flipped')
  local destFolderFlipped = tiefvision_commons.dataPath('db/similarity/img-flipped-similarity-deeprank')
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
  torch.save(destFolder .. '/' .. imageName, encodedImage)
end

function loadModel()
  return torch.load(tiefvision_commons.modelPath('similarity.model'))
end

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
