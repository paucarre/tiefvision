-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua;../11-similarity-searcher-cnn-db/?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local tiefvision_commons = require 'tiefvision_commons'
local search_commons = require 'search_commons'

function getTestError(reference)
  local dataFolder = '../data/db/similarity/img-enc-cnn-encoder'
  local similarityDb = '../data/db/similarity/img-unsup-similarity-db'
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

local reference = arg[1]
getTestError(reference)
