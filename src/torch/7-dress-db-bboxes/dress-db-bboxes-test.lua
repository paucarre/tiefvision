-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;../6-bboxlib/?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local image = require 'image'
tiefvision_commons = require 'tiefvision_commons'
bboxlib = require 'bboxlib'

local folder = "../../../resources/dresses-db/master"
local bboxesFolder = "../../../resources/dresses-db/bboxes"
local files = {"1069838451.jpg" }
for fileIndex = 1, #files do
  print(files[fileIndex])
  local fileName = folder .. '/' .. files[fileIndex]
  local input = bboxlib.loadImageFromFile(fileName)
  local bboxes = bboxlib.getImageBoundigBoxesTable(input, 1)
  local imageBoundingBoxes = bboxlib.getImageBoundingBoxesMap(bboxes, probs, imageW, imageH, reduction, scale, input)
  for i = 1, #bboxes do
    local xmin = bboxes[i][1]
    local ymin = bboxes[i][2]
    local xmax = bboxes[i][3]
    local ymax = bboxes[i][4]
    local inputCropped = image.crop(input, xmin, ymin, xmax, ymax)
    image.save(bboxesFolder .. '/' .. i .. '/' .. files[fileIndex], inputCropped)
  end
end

print('Bounding Box Dresses DB Generated')
