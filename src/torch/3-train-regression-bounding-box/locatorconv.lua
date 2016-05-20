-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

require "inn"
require 'optim'
require 'torch'
require 'xlua'

local locatorconv = {}

function locatorconv.loadModel()
  local nhiddens1 = 512
  local nhiddens2 = 128
  local noutputs = 1
  local model = nn.Sequential()
  model:add(nn.SpatialConvolutionMM(384, nhiddens1, 11, 11, 1, 1, 0, 0))
  model:add(nn.Tanh())
  model:add(nn.SpatialConvolutionMM(nhiddens1, nhiddens2, 1, 1, 1, 1, 0, 0))
  model:add(nn.Tanh())
  model:add(nn.SpatialConvolutionMM(nhiddens2, noutputs, 1, 1, 1, 1, 0, 0))
  return model:cuda()
end

return locatorconv
