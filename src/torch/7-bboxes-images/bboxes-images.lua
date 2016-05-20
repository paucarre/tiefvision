-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;../6-bboxlib/?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local image = require 'image'
local tiefvision_commons = require 'tiefvision_commons'
local bboxlib = require 'bboxlib'

function getFiles(folder)
  local files = {}
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file, "mode") == "file") then
      table.insert(files, file)
    end
  end
  return files
end

local folder = "../../../resources/dresses-db/master"
local bboxesFolder = "../../../resources/dresses-db/bboxes"
local flippedBboxesFolder = "../../../resources/dresses-db/bboxes-flipped"
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
