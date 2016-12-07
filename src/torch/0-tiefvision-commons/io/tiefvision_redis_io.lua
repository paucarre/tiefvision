-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Reader and writer to store information thanks to redis
--

local paths = require('paths')
local torchFolder = paths.thisfile('../..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local redis = require 'redis'

local function removeDatabaseFromKey(key, database)
  -- redis key format is "database:filename"
  -- to remove database, sub requires database length + 1
  -- to remove the column, sub requires + 1
  return string.sub(key, #database + 2)
end

local function toRedisProtocol(...)
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

local tiefvision_redis_io = {}
function tiefvision_redis_io.read(database, key)
  local response = tiefvision_redis_io.redisClient:hgetall(database .. ':' .. key)
  local responseWithNewKeys = {}
  for k, v in pairs(response) do
    local newKey = removeDatabaseFromKey(k, database)
    responseWithNewKeys[newKey] = v
  end

  return responseWithNewKeys
end

function tiefvision_redis_io.write(database, key, value)
  local tmpFileName = paths.tmpname()
  local file = io.open(tmpFileName, "w")

  file:write(toRedisProtocol("DEL", database .. ':' .. key))
  for k, v in pairs(value) do
    file:write(toRedisProtocol("HSET", database .. ':' .. key, k, v))
  end

  file:close()
  os.execute("cat " .. tmpFileName .. " | redis-cli --pipe -h " .. tiefvision_redis_io.host .. " -p " .. tiefvision_redis_io.port .. " 1>/dev/null &")
end

function tiefvision_redis_io.keys(database)
  local keys = tiefvision_redis_io.redisClient:keys(database .. ":*")
  for i = 1, #keys do
    keys[i] = removeDatabaseFromKey(keys[i], database)
  end

  return keys
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
