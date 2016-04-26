-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;../5-train-classification/?.lua;./?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
require 'image'
local tiefvision_commons = require 'tiefvision_commons'
local similarity_db_lib = require 'similarity_db_lib'

function createDb(sourceFolder, destinationFolder)
  local files = tiefvision_commons.getFiles(sourceFolder)
  local encoder = similarity_db_lib.getEncoder()
  for fileIndex = 1, #files do
    local file = files[fileIndex]
    local destPath = destinationFolder .. '/' .. file
    if(not tiefvision_commons.fileExists(destPath)) then
      print('Encoding ' .. file)
      local encoderOutput = similarity_db_lib.encodeImage(sourceFolder .. '/' .. file, encoder)
      torch.save(destPath, encoderOutput)
      collectgarbage()
    end
  end
end

createDb("../../../resources/dresses-db/bboxes/1", "../data/encoded-images")
createDb("../../../resources/dresses-db/bboxes-flipped/1", "../data/encoded-images-flipped")
