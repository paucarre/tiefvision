/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
  */
package controllers

import com.google.inject.Inject
import play.api.{Play, Environment}

class TiefVisionResourcesAssets @Inject()(environment: Environment) extends ExternalAssets {

  def atResources(rootPath: String, file: String) = super.at(s"${Configuration.HomeFolder}/$rootPath", file)

}
