-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'inn'
require 'optim'
require 'xlua'
require 'lfs'
local torch = require 'torch'

local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'
local similarity_lib = require '14-deeprank-db/similarity_lib'
local database = require('0-tiefvision-commons/tiefvision_config_loader').load().database

local function similarityDb()
  local similaritiesDb = 'image-supervised-similarity-database'

  local dataFolder = tiefvision_commons.dataPath('db/similarity/img-similarity-deeprank')
  local files = tiefvision_commons.getFiles(dataFolder)
  local filesAlreadyProcessed = database.keys(similaritiesDb)
  local filesRemaining = tiefvision_commons.tableSubtraction(files, filesAlreadyProcessed)

  for referenceIndex = 1, #filesRemaining do
    local reference = filesRemaining[referenceIndex]
    print(reference)

    local similarities = {}

    local referenceEncoding = torch.load(dataFolder .. '/' .. reference):double()
    for testIndex = 1, #files do
      local test = files[testIndex]
      local imageEncoding = torch.load(dataFolder .. '/' .. test):double()
      local similarity = similarity_lib.similarity(referenceEncoding, imageEncoding)
      similarities[test] = similarity or -1
    end

    -- compare itself with its mirror
    local flipped = tiefvision_commons.dataPath('db/similarity/img-flipped-similarity-deeprank', reference)
    local flippedEncoding = torch.load(flipped):double()
    local similarity = similarity_lib.similarity(referenceEncoding, flippedEncoding)
    similarities[reference] = similarity or -1

    database.write(similaritiesDb, reference, similarities)

    if referenceIndex % 10 == 0 then
      collectgarbage()
    end
  end
end

similarityDb()
