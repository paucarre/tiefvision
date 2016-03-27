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
   local dataFolder = '../data/db/similarity/img-enc-cnn-encoder'
   local testLines = tiefvision_commons.getFiles(dataFolder)
   local referenceEncoding = torch.load(dataFolder .. '/' ..  reference):double()
   local comparisonTable = {}
   for testIndex = 1, #testLines do
     local file = testLines[testIndex]
     local imageEncoding = torch.load(dataFolder .. '/' .. file):double()
     local sumDist = 0.0
     local minHeight = math.min(referenceEncoding:size()[2], imageEncoding:size()[2])
     local maxHeight = math.max(referenceEncoding:size()[2], imageEncoding:size()[2])
     if( maxHeight - minHeight < 5) then
       for w = 1, referenceEncoding:size()[1] do
         for h = 1, minHeight do
           local distLoc = getAngle(imageEncoding[w][h],  referenceEncoding[w][h])
           sumDist =  sumDist + distLoc
         end
       end
      local dist = sumDist / (referenceEncoding:size()[1] * minHeight)
      table.insert(comparisonTable, {file, dist, distMean})
      end
   end
   table.sort(comparisonTable, sortCmpTable)
   printCmpTable(comparisonTable)
end

function getDist(a, b)
  return torch.norm(a - b)
end

function getAngle(a, b)
  -- local divi = torch.norm(a) * torch.norm(b)
  local angle = math.abs(math.acos(a * b))
  if(isnan(angle)) then
    return 0.0
  else
    return angle
  end
end

function isnan(x) return x ~= x end


function sortCmpTable(a, b)
  return a[2] < b[2]
end

function printCmpTable(cmpTable)
  for i = 1, #cmpTable do
     -- print(cmpTable[i][1] .. ' ' .. cmpTable[i][2])
    print(cmpTable[i][1] .. ' ' .. cmpTable[i][2] .. ' ' .. cmpTable[i][3] )
  end
end

local reference = arg[1]
getTestError(reference)
