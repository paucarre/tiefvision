-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

package.path = package.path .. ';../0-tiefvision-commons/?.lua'
require 'inn'
require 'optim'
require 'torch'
require 'xlua'
require 'lfs'
local tiefvision_commons = require 'tiefvision_commons'

function getTestError(reference)
  local similarityDb = '../data/db/similarity/img-sup-similarity-db'
  local dataFolder = '../data/db/similarity/img-similarity-deeprank'
  local testLines = tiefvision_commons.getFiles(dataFolder)
  local similarities = torch.load(similarityDb):double()
  local referenceIndex = getIndex(testLines, reference)
  local comparisonTable = {}
  for testIndex = 1, #testLines do
    local file = testLines[testIndex]
    local sim = similarities[referenceIndex][testIndex]
    table.insert(comparisonTable, { file, sim })
  end
  table.sort(comparisonTable, sortCmpTable)
  printCmpTable(comparisonTable)
end

function getIndex(testLines, reference)
  for testIndex = 1, #testLines do
    if (testLines[testIndex] == reference) then
      return testIndex
    end
  end
  return 0
end

function sortCmpTable(a, b)
  return a[2] > b[2]
end

function printCmpTable(cmpTable)
  for i = 1, #cmpTable do
    print(cmpTable[i][1] .. ' ' .. cmpTable[i][2])
  end
end

local reference = arg[1]
getTestError(reference)
