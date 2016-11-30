-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

local search_commons = {}

function search_commons.sortCmpTable(a, b)
  return a[2] > b[2]
end

function search_commons.printCmpTable(cmpTable)
  for i = 1, #cmpTable do
    print(cmpTable[i][1] .. ' ' .. cmpTable[i][2])
  end
end

return search_commons
