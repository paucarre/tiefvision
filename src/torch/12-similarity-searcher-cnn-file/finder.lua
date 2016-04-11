-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;../6-bboxlib/?.lua;../11-similarity-db/?.lua;../9-similarity-db-cnn/?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local tiefvision_commons = require 'tiefvision_commons'
local bboxlib = require 'bboxlib'
local similarity_lib = require 'similarity_lib'
local similarity_db_lib = require 'similarity_db_lib'

function getTestError(referenceEncoding)
   local dataFolder = '../data/db/similarity/img-enc-cnn-encoder'
   local testLines = tiefvision_commons.getFiles(dataFolder)
   local comparisonTable = {}
   for testIndex = 1, #testLines do
     local file = testLines[testIndex]
     local imageEncoding = torch.load(dataFolder .. '/' .. file):double()
     local dist = similarity_lib.similarity(referenceEncoding, imageEncoding)
     table.insert(comparisonTable, {file, dist})
   end
   table.sort(comparisonTable, sortCmpTable)
   printCmpTable(comparisonTable)
end

function sortCmpTable(a, b)
  return a[2] > b[2]
end

function printCmpTable(cmpTable)
  for i = 1, #cmpTable do
    print(cmpTable[i][1] .. ' ' .. cmpTable[i][2])
  end
end

function getImage(fileName)
  local input = bboxlib.loadImageFromFile('../../../resources/dresses-db/uploaded/master/' .. fileName)
  local bboxes = bboxlib.getImageBoundingBoxesTable(input, 1)
  local xmin = bboxes[1][1]
  local ymin = bboxes[1][2]
  local xmax = bboxes[1][3]
  local ymax = bboxes[1][4]
  input = image.crop(input, xmin, ymin, xmax, ymax)
  image.save('../../../resources/dresses-db/uploaded/bbox/' .. fileName, input)
  local encoder = similarity_db_lib.getEncoder()
  local encodedOutput = similarity_db_lib.encodeImage('../../../resources/dresses-db/uploaded/bbox/' .. fileName, encoder)
  return encodedOutput:double()
end

local fileName = arg[1]
local referenceDress = getImage(fileName)
getTestError(referenceDress)
