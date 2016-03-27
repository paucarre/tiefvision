-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua'
require 'nn'
require 'inn'
require 'image'
tiefvision_commons = require 'tiefvision_commons'
require'lfs'

function getFiles()
  local files = {}
  local folder = "../../../resources/regression-gown-bounding-boxes/only_gown/scaled"
  for file in lfs.dir(folder) do
    if (lfs.attributes(folder .. '/' .. file,"mode") == "file") then
      table.insert(files, file)
    end
  end
  return files
end

function encodeData()
  local encoder = torch.load('../models/encoder.model')
  local folder = "../../../resources/regression-gown-bounding-boxes/only_gown/scaled"
  local files = getFiles()
  for i = 1, #files do
    print(files[i])
    local input = tiefvision_commons.load(folder .. '/' .. files[i])

    -- top layer
    local highInput = encoder:forward(input)[1]
    saveNormalized(highInput, 'high/' .. files[i])

    -- lower layer
    local lowModule = encoder.modules[7]
    local lowInput  = lowModule.output
    saveNormalized(lowInput, 'low/' .. files[i])
  end
end

function saveNormalized(encodedInput, file)
  local normEncodedInput = encodedInput
  -- / torch.norm(encodedInput)
  local transposedNormEncodedInput = normEncodedInput:transpose(1, 3)
  -- local closeToOne = torch.norm(normEncodedInput)
  -- assert(math.abs(closeToOne - 1.0) < 0.001, "the input should be normalized")
  torch.save('../data/db/' .. file, transposedNormEncodedInput)
end

encodeData()

print('DB encoding finished')
