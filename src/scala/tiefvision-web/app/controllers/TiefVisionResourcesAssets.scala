/**
  * Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
  * You may use, distribute and modify this code under the
  * terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).
  */
package controllers

import java.io.File
import com.google.inject.Inject
import play.api.{Environment, Logger, Play}
import play.api.mvc._

class TiefVisionResourcesAssets @Inject()(environment: Environment) extends Controller {

  // Request local resources within the TIEFVISION_HOME folder (for security reasons)
  def atResources(rootPath: String, filePath: String) = Action {
    val file = new File(s"${Configuration.HomeFolder}/$rootPath/$filePath")
    val fileAbsolutePath = file.getAbsolutePath()

    if(fileAbsolutePath.startsWith(Configuration.HomeFolder)) {
      Ok.sendFile(file)
    } else {
      Logger.error(s"Dangerous file requested: ${fileAbsolutePath} !!!")
      NotFound
    }
  }

}
