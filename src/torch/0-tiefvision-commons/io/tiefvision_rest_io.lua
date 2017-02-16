-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Reader and writer to store information thanks to redis
--

local paths = require('paths')
local torchFolder = paths.thisfile('../..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local cjson = require("cjson")
local curl = require('0-tiefvision-commons/io/tiefvision_curl_io')

local factory = {}
setmetatable(factory, { __call = function(_, url, fileJsonKey, scoreJsonKey)
  local urlKeys = function() return url end
  local urlRead = function(key) return url .. "/" .. key end
  local urlWrite = function(key) return url .. "/" .. key end

  local responseToKeys = function(response)
    return cjson.decode(response)
  end

  local responseToValue = function(response)
    local values = {}
    for _, value in pairs(cjson.decode(response)) do
      local file = value[fileJsonKey]
      local score = value[scoreJsonKey]

      values[file] = score
    end

    return values
  end

  local valueToRequest = function(value)
    local buff = {}
    for file, score in pairs(value) do
      local json = string.format('{"%s":%s,"%s":%s}', fileJsonKey, file, scoreJsonKey, score)
      table.insert(buff, json)
    end

    return {
      httpheader = { "Content-Type: application/json" },
      postfields = string.format('[%s]', table.concat(buff, ","))
    }
  end

  return curl(urlKeys, responseToKeys, urlRead, responseToValue, urlWrite, valueToRequest)
end })

return factory
