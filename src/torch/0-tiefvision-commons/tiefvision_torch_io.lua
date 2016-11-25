-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Reader and writer to store information thanks to torch
--

local paths = require('paths')
local torchFolder = paths.thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local torch = require 'torch'

local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'
local tiefvision_torch_io = {}

function filePath(fileName)
  return tiefvision_commons.dataPath(fileName)
end

function tiefvision_torch_io.read(fileName)
  local file = filePath(fileName)
  if not paths.filep(file) then
    return nil
  end

  return torch.load(file)
end

function tiefvision_torch_io.write(fileName, data)
  local file = filePath(fileName)
  torch.save(file, data)
end

return tiefvision_torch_io
