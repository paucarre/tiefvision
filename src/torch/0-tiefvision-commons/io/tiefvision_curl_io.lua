-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Reader and writer to store information thanks to redis
--

local paths = require('paths')
local torchFolder = paths.thisfile('../..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local cURL = require "cURL"

function performCurlRequest(curl)
  local buff = {}
  curl:setopt_writefunction(function(data) table.insert(buff, data) end)
    :perform()

  local response_code = curl:getinfo_response_code()
  local response = table.concat(buff)

  curl:close()

  if response_code ~= 200 then
    print(response)
    os.exit(1)
  end

  return response
end

local tiefvision_curl_io = {}

function tiefvision_curl_io.keys()
  local request = cURL.easy { url = tiefvision_curl_io.urlKeys() }
  local response = performCurlRequest(request)
  local keys = tiefvision_curl_io.responseToKeys(response)

  return keys
end

function tiefvision_curl_io.read(key)
  local request = cURL.easy { url = tiefvision_curl_io.urlRead(key) }
  local response = performCurlRequest(request)
  local value = tiefvision_curl_io.responseToValue(response)

  return value
end

function tiefvision_curl_io.write(key, value)
  local requestObj = { url = tiefvision_curl_io.urlWrite(key) }
  for k,v in pairs(tiefvision_curl_io.valueToRequest(value)) do requestObj[k] = v end

  local request = cURL.easy(requestObj)

  performCurlRequest(request)
end

local factory = {}
setmetatable(factory, { __call = function(_, urlKeys, responseToKeys, urlRead, responseToValue, urlWrite, valueToRequest)
  tiefvision_curl_io.urlKeys = urlKeys
  tiefvision_curl_io.responseToKeys = responseToKeys

  tiefvision_curl_io.urlRead = urlRead
  tiefvision_curl_io.responseToValue = responseToValue

  tiefvision_curl_io.urlWrite = urlWrite
  tiefvision_curl_io.valueToRequest = valueToRequest

  return tiefvision_curl_io
end })

return factory
