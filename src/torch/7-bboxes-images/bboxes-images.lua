-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'

local image = require 'image'
local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'
local bboxlib = require '6-bboxlib/bboxlib'

function getFiles(folder)
  local files = {}
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file, "mode") == "file") then
      table.insert(files, file)
    end
  end
  return files
end

local folder = tiefvision_commons.resourcePath('dresses-db/master')
local bboxesFolder = tiefvision_commons.resourcePath('dresses-db/bboxes')
local flippedBboxesFolder = tiefvision_commons.resourcePath('dresses-db/bboxes-flipped')
local files = getFiles(folder)
for fileIndex = 1, #files do
  if not tiefvision_commons.fileExists(bboxesFolder .. '/1/' .. files[fileIndex]) then
    print(files[fileIndex])
    local fileName = folder .. '/' .. files[fileIndex]
    local input = bboxlib.loadImageFromFile(fileName)
    local bboxes = bboxlib.getImageBoundingBoxesTable(input, 1)
    for i = 1, #bboxes do
      local xmin = bboxes[i][1]
      local ymin = bboxes[i][2]
      local xmax = bboxes[i][3]
      local ymax = bboxes[i][4]
      print(xmin, ymin, xmax, ymax)
      local inputCropped = image.crop(input, xmin, ymin, xmax, ymax)
      image.save(bboxesFolder .. '/' .. i .. '/' .. files[fileIndex], inputCropped)
      -- generate flipped images 
      local flippedInput = image.hflip(inputCropped)
      image.save(flippedBboxesFolder .. '/' .. i .. '/' .. files[fileIndex], flippedInput)
      collectgarbage()
    end
  end
end

print('Bounding Box Dresses DB Generated')
