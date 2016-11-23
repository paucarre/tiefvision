-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Commmon utility methods that are used thoughout the other modules. They are mostly
-- related to IO.
--

local image = require 'image'
local lfs = require 'lfs'
local tiefvision_commons = {}

function tiefvision_commons.fileExists(name)
  local f = io.open(name, "r")
  if f ~= nil then io.close(f) return true else return false end
end

function tiefvision_commons.getLines(filename)
  local trainFile = io.open(filename)
  print(trainFile)
  local lines = {}
  if trainFile ~= nil then
    local index = 1
    for trainFileLine in trainFile:lines() do
      print(trainFileLine)
      if (tiefvision_commons.fileExists(trainFileLine)) then
        lines[index] = trainFileLine
        index = index + 1
      end
    end
    io.close(trainFile)
  end
  return lines
end

function tiefvision_commons.getFiles(folder)
  local files = {}
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file, "mode") == "file") then
      table.insert(files, file)
    end
  end
  return files
end

-- Loads the mapping from net outputs to human readable labels
function tiefvision_commons.load_synset()
  local file = io.open 'synset_words.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line, 11))
  end
  io.close(file)
  return list
end

function tiefvision_commons.img_mean()
  local img_mean_name = tiefvision_commons.modelPath('ilsvrc_2012_mean.t7')
  return torch.load(img_mean_name).img_mean:transpose(3, 1)
end

function tiefvision_commons.load(imagePath)
  local img = image.load(imagePath)
  img = tiefvision_commons.preprocess(img)
  return img
end

function tiefvision_commons.loadImage(img)
  img = tiefvision_commons.preprocess(img)
  return img
end

function tiefvision_commons.preprocess(im)
  local img_mean = tiefvision_commons.img_mean()
  local scaledImage = im * 255
  -- converts RGB to BGR
  local bgrImage = scaledImage:clone()
  bgrImage[{ 1, {}, {} }] = scaledImage[{ 3, {}, {} }]
  bgrImage[{ 3, {}, {} }] = scaledImage[{ 1, {}, {} }]

  local imageMinusAvg = bgrImage - image.scale(img_mean, im:size()[2], im:size()[3], 'bilinear')
  return imageMinusAvg:cuda()
end

function tiefvision_commons.path(...)
  local environmentVariableName = 'TIEFVISION_HOME'
  local root = os.getenv(environmentVariableName)
  if not root then
    error(string.format("missing environment variable: %s", environmentVariableName))
  end

  local file_path_table = {...}
  table.insert(file_path_table, 1, root)
  local file_path, _ = table.concat(file_path_table, '/'):gsub('/+', '/')
  return file_path
end

function tiefvision_commons.dataPath(...)
  return tiefvision_commons.path('src/torch/data', unpack({...}))
end

function tiefvision_commons.modelPath(...)
  return tiefvision_commons.path('src/torch/models', unpack({...}))
end

function tiefvision_commons.resourcePath(...)
  return tiefvision_commons.path('resources', unpack({...}))
end

return tiefvision_commons
