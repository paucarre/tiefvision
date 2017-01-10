-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

--
-- Default configuration file
--

local torchFolder = require('paths').thisfile('.')
package.path = string.format("%s;%s/?.lua", os.getenv("LUA_PATH"), torchFolder)

local tiefvision_commons = require('0-tiefvision-commons/tiefvision_commons')
local database_factory = require('0-tiefvision-commons/io/tiefvision_torch_io')

return {
  database = {
    supervised_similarity = database_factory(tiefvision_commons.dataPath('image-supervised-similarity-database')),
    unsupervised_similarity = database_factory(tiefvision_commons.dataPath('image-unsupervised-similarity-database'))
  }
}
