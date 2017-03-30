-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Get k nearest neighbor
--

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local tiefvision_reduction = {}

tiefvision_reduction.getNearestNeighbors = function(neighbors, k)
    local keysSortedByValue = function(tbl, sortFunction)
        local keys = {}
        for key in pairs(tbl) do table.insert(keys, key) end
        table.sort(keys, function(a, b) return sortFunction(tbl[a], tbl[b]) end)

        return keys
    end

    local neighborsKey = keysSortedByValue(neighbors, function(a, b) return a > b end)
    local kMin = math.min(#neighborsKey, k)

    local buff = {}
    for i = 1, kMin do
        local key = neighborsKey[i]
        local value = neighbors[key]

        buff[key] = value
    end

    return buff
end

return tiefvision_reduction
