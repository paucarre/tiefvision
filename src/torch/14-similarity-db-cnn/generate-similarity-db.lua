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

function createDb()
  local folder = "../../../resources/dresses-db/bboxes/1"
  local destFolder = "../data/db/similarity/img-enc-cnn-encoder"
  local files = tiefvision_commons.getFiles(folder)
  local encoder = similarity_db_lib.getEncoder()
  for fileIndex = 1, #files do
    local file = files[fileIndex]
    local encoderOutput = similarity_db_lib.encodeImage(folder .. '/' .. file, encoder)
    print('Encoded ' .. file)
    torch.save(destFolder .. '/' .. file, encoderOutput)
    collectgarbage()
  end
end

createDb()
