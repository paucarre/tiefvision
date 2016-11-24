-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- It Splits the (Alexnet-like) network into an encoder and a classifier
-- The classifier is discarded and the encoder is used by other modules
-- to encode images.
--

local torchFolder = require('paths').thisfile('..')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

require 'loadcaffe'
require 'image'
require 'inn'

local tiefvision_commons = require '0-tiefvision-commons/tiefvision_commons'

local proto_name = 'deploy.prototxt'
local model_name = 'nin_imagenet.caffemodel'
local image_name = 'Goldfish3.jpg'

local net = loadcaffe.load(proto_name, './nin_imagenet.caffemodel'):cuda()
net.modules[#net.modules] = nil -- remove the top softmax

net:evaluate()
local synset_words = tiefvision_commons.load_synset()

local im = tiefvision_commons.load(image_name)

local loss, output = net:forward(im):view(-1):float():sort(true)

-- create classification encoder
local encoder = net:clone()
for i = 1, 9 do
  encoder:remove(21)
end
local concat = nn.ConcatTable()
concat:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil():cuda())
concat:add(nn.SpatialMaxPooling(3, 3, 1, 1, 0, 0):ceil():cuda())
encoder:add(concat)
encoder:evaluate()
print(encoder)

-- create classifier
local classifier = net:clone()
for i = 1, 21 do
  classifier:remove(1)
end

local outputEnc = encoder:forward(im)
local lossClassifier, outputClassifier = classifier:forward(outputEnc[1]):view(-1):float():sort(true)

local outputSize = output:size()[1]
assert(outputSize == 1000, 'Output size should be 1000')
assert(torch.eq(outputClassifier, output), 'the output of the network should be the same as the output of the classifier using the encoder')

-- test expected output
local expectedOutput = torch.Tensor(4)
expectedOutput[1] = 2
expectedOutput[2] = 89
expectedOutput[3] = 91
expectedOutput[4] = 131
for i = 1, expectedOutput:size()[1] do
  print(outputClassifier[i] .. ' ' .. expectedOutput[i] .. ' ' .. synset_words[outputClassifier[i]] .. ' || ' .. synset_words[output[i]])
  assert(outputClassifier[i] == expectedOutput[i], 'the network predicted an unexpected class')
end

-- Test encoder reduction
local fakeIm = torch.Tensor(3, 224, 224):cuda()
local outputEnc = encoder:forward(fakeIm)
assert(outputEnc[1]:size()[3] == 6, 'the encoder size for 224 input size should be 6')
assert(outputEnc[2]:size()[3] == (outputEnc[1]:size()[3] * 2) - 1, 'the regression encoder size for 2240 input size should be the double of the classification one')

fakeIm = torch.Tensor(3, 224, 224 * 10):cuda()
local outputEnc = encoder:forward(fakeIm)
assert(outputEnc[1]:size()[3] == 69, 'the encoder size for 2240 input size should be 69')
assert(outputEnc[2]:size()[3] == (outputEnc[1]:size()[3] * 2) - 1, 'the regression encoder size for 2240 input size should be the double of the classification one')

fakeIm = torch.Tensor(3, 224, 224 * 15):cuda()
local outputEnc = encoder:forward(fakeIm)
assert(outputEnc[1]:size()[3] == 104, 'the encoder size for 4480 input size should be 104')
assert(outputEnc[2]:size()[3] == (outputEnc[1]:size()[3] * 2) - 1, 'the regression encoder size for 224 * 15 input size should be the double of the classification one')

print("Saving models...")

torch.save(tiefvision_commons.modelPath('net.model'), net)
torch.save(tiefvision_commons.modelPath('encoder.model'), encoder)
torch.save(tiefvision_commons.modelPath('classifier-original.model'), classifier)

print("Finished!")
