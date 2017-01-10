-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Get config object from cli, env, or a default
--

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local tiefvision_commons = require('0-tiefvision-commons/tiefvision_commons')

local function argument(arg)
  local prefix = '^%-%-?config'
  local prefixAndEqual = prefix .. '='

  for index, value in pairs(arg) do
    if string.match(value, prefixAndEqual) then
      return (string.gsub(value, prefixAndEqual, ''))
    elseif string.match(value, prefix) then
      return arg[index + 1]
    end
  end

  return nil
end

local configLoader = {}

configLoader.argument = argument(arg)
configLoader.environment = os.getenv('CONFIG')
configLoader.default = tiefvision_commons.path(
  tiefvision_commons.rootPath(),
  'src/torch/config.lua')

configLoader.file = configLoader.argument or
  configLoader.environment or
  configLoader.default

function configLoader.load()
  package.path = package.path .. ';' .. configLoader.file
  return require(configLoader.file)
end

if not require('paths').filep(configLoader.file) then
  error("configuration file doesn't exist: " .. configLoader.file)
end

return configLoader
