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
local similarity_lib = require '9-similarity-db/similarity_lib'
local tiefvision_config_loader = require('0-tiefvision-commons/tiefvision_config_loader')
local database = tiefvision_config_loader.load().database

function similarityDb(imageFolder)
  local similaritiesDb = 'image-unsupervised-similarity-database'

  local files = tiefvision_commons.getFiles(imageFolder)
  local filesAlreadyProcessed = database.keys(similaritiesDb)
  local filesRemaining = tiefvision_commons.tableSubtraction(files, filesAlreadyProcessed)

  for referenceIndex = 1, #filesRemaining do
    local reference = filesRemaining[referenceIndex]
    print(reference)

    local similarities = {}

    local referenceEncoding = torch.load(imageFolder .. '/' .. reference):double()
    for testIndex = 1, #files do
      local test = files[testIndex]
      local imageEncoding = torch.load(imageFolder .. '/' .. test):double()
      local similarity = similarity_lib.similarity(referenceEncoding, imageEncoding)
      similarities[test] = similarity or -1
    end

    database.write(similaritiesDb, reference, similarities)

    if referenceIndex % 5 == 0 then
      collectgarbage()
    end
  end
end

function getOptions()
  local cmd = torch.CmdLine()
  cmd:text('Compare images to one another to identify which are the most similar')
  cmd:text('Options:')
  cmd:option('-images', tiefvision_commons.dataPath('encoded-images'), 'Directory to load images')
  cmd:text('')
  cmd:option('-config', tiefvision_config_loader.default, 'Configuration file to use.')

  return cmd:parse(arg)
end

local options = getOptions()
similarityDb(options.images)
