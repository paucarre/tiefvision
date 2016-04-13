-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

local search_commons = {}

function search_commons.getIndex(testLines, reference)
  for testIndex = 1, #testLines do
    if (testLines[testIndex] == reference) then
      return testIndex
    end
  end
  return 0
end

function search_commons.sortCmpTable(a, b)
  return a[2] > b[2]
end

function search_commons.printCmpTable(cmpTable)
  for i = 1, #cmpTable do
    print(cmpTable[i][1] .. ' ' .. cmpTable[i][2])
  end
end

return search_commons


