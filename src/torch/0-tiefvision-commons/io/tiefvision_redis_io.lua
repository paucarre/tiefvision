-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Reader and writer to store information thanks to redis
--

local paths = require('paths')
local torchFolder = paths.thisfile('../..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local torch = require 'torch'
local redis = require 'redis'
local redisClient = nil

local tiefvision_redis_io = {}
function tiefvision_redis_io.read(fileName)
  local key = toKey(fileName)
  local response = tiefvision_redis_io.redisClient:hgetall(key)

  local responseInt = {}
  for key, value in pairs(response) do
    local keyInt = tonumber(key)
    responseInt[keyInt] = value
  end

  return responseInt
end

function tiefvision_redis_io.write(fileName, data)
  local key = toKey(fileName)

  local tmpFileName = paths.tmpname()
  local file = io.open(tmpFileName, "w")

  file:write(toRedisProtocol("DEL", key))
  for i = 1, data:size()[1] do
    file:write(toRedisProtocol("HSET", key, i, data[i]))
  end

  file:close()
  os.execute("cat " .. tmpFileName .. " | redis-cli --pipe -h " .. tiefvision_redis_io.host .. " -p " .. tiefvision_redis_io.port .. " 1>/dev/null &")
end

function tiefvision_redis_io.last(fileName, data)
  local keys = tiefvision_redis_io.redisClient:keys("*")
  local keysInt = {}
  for i = 1, #keys do
    keysInt[i] = tonumber(keys[i])
  end

  return (keysInt[#keysInt] or 0) + 1
end

function toKey(fileName)
  return string.gsub(fileName, ".*/", "")
end

function toRedisProtocol(...)
  local args = {...}
  local argsLength = #args

  local redisProtocol = "*" .. argsLength .. "\r\n"
  for i = 1, argsLength do
    local arg = tostring(args[i])

    redisProtocol = redisProtocol .. "$" .. #arg .. "\r\n"
    redisProtocol = redisProtocol .. arg .. "\r\n"
  end

  return redisProtocol
end

local factory = {}
setmetatable(factory, { __call = function(_, host, port)
  tiefvision_redis_io.host = host
  tiefvision_redis_io.port = port or 6379

  tiefvision_redis_io.redisClient = redis.connect(
    tiefvision_redis_io.host,
    tiefvision_redis_io.port
  )

  return tiefvision_redis_io
end })

return factory
