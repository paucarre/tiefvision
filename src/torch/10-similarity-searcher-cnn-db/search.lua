-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local libsFolder = require('paths').thisfile('..')
package.path = package.path .. ';' .. libsFolder .. '/0-tiefvision-commons/?.lua'

require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local tiefvision_commons = require 'tiefvision_commons'
local search_commons = require 'search_commons'

function getTestError(reference)
  local dataFolder = tiefvision_commons.dataPath('encoded-images')
  local similarityDb = tiefvision_commons.dataPath('img-unsup-similarity-db')
  local testLines = tiefvision_commons.getFiles(dataFolder)
  local similarities = torch.load(similarityDb):double()
  local referenceIndex = search_commons.getIndex(testLines, reference)
  local comparisonTable = {}
  for testIndex = 1, #testLines do
    local file = testLines[testIndex]
    local sim = similarities[referenceIndex][testIndex]
    table.insert(comparisonTable, { file, sim })
  end
  table.sort(comparisonTable, search_commons.sortCmpTable)
  search_commons.printCmpTable(comparisonTable)
end

function getOptions()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Unsupervised image search from precomputed database of distances between each pair of images in the master folder.')
  cmd:text('Returns a descending sorted list of filenames concatenated with a similarity metric.')
  cmd:text('Both the filename to search and the result filenames come from the folder $TIEFVISION_HOME/resources/dresses-db/master.')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-image', 'Filename (not full path, just the filename) from $TIEFVISION_HOME/resources/dresses-db/master.')
  cmd:text()
  local options = cmd:parse(arg)
  if(options.image == nil) then
    cmd:help()
    os.exit()
  end
  return options
end

local options = getOptions()
getTestError(options.image)
