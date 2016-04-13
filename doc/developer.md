## Developer Guide

This document explains the different modules in **TiefVision** and how to use them.
Even though it's developer oriented, it can also be used as a user manual.

### Requirements

The current mandatory requirements to make **TiefVision** work are the following:

* Development machine with an **nVidia CUDA** graphics card.
Note that there so far no will to remove this requirement. I might move it to **OpenCL** at the time **Torch** fully supports it for all neural network layers, it performs as fast as **CUDA** and it's really mature.
* **Linux OS** ( other Unix-like OSs *should* also work)
* Latest version of **Torch**
* Compatible **Java Development Kit 8** (recommended Oracle (Copyright) JDK )
* An **H2** database server. I don't think **TiefVision** will ever support another database.

## Environment setup

Once your machine is set up with all the required software and hardware, you'll
have to follow these steps:

* Clone TiefVision GIT repository:
```
git clone git@github.com:paucarre/tiefvision.git
```
* Create the *TIEFVISION_HOME* envrionment variable pointing to your cloned repository and add it into your bash profile.
```
export TIEFVISION_HOME=<PATH_TO_TIEFVISION_REPOSITORY>
```

**TODO**: setup of H2 database

**TODO**: Transfer learning for the encoder

## Bounding Box Regression

The first step in **TiefVision** is to detect where in the image you target object is located. For example, in the case of a dress style search engine, what we would be interested is on predicting the top-left and bottom-right corners of the bounding box that encompasses the dress. For a face search engine, you would a network able to detect the bounding box that encompasses the face in the image.

First off, you'll have to first download a set of images and place them in
**<$TIEFVISION_HOME>/resources/dresses-db/master**. There shouldn't be any major constraint regarding the image format but it's only tested with **JPG** and therefore that's the only supported format.

Keep in mind that the file name will identify the image and therefore it's wise
to name the image using some sort of database identifier so you can later on keep track of the image and the product it's related to.

The next step is to create bounding boxes for those images. That's a pretty annoying task but at least with **TiefVision** you have a very basic web bounding box editor that'll make your life easier. To do that you'll have to start both the web server and also the **H2** database server.

To start the web server do the following:
```
cd $TIEFVISION_HOME/src/scala/tiefvision-web
activator run
```

To start the H2 database sever do the following:
```
java -cp <PATH_TO_H2_JAR> org.h2.tools.Server
```

Once you are done open a web browser and open the [bounding box endpoint](http://localhost:9000/bounding_box).
There you'll be able to create bounding boxes and store them into the database.
Make sure after the your save the first bounding box that there is no error log in the web server console and that the entry is stored in the database by running
the following SQL query:
```sql
SELECT COUNT(*) FROM BOUNDING_BOX
```
This query must output the number of bounding boxes that are stored in the database.
If the result of the query is zero and you are supposed to  have saved a bounding box, there is a problem as nothing has been saved (check the logs!).

Once you have already tagged enough bounding boxes (say, around 1.000) you'll need to generate the training and the test set to train the neural network. That's something that **TiefVision** automatically does for you by opening the [Bounding Box Test and Training endpoint](http://localhost:9000/generate_bounding_box_train_and_test_files). That endpoint will generate crops that have at least 50% of its area within the bounding box of the target object. It'll also generate training and test sets in such a way all the crops in the training set come from images that don't belong to the images from the test set crops.   

The crops are generated in  **<$TIEFVISION_HOME>/resources/bounding-boxes/crops**. The training set file is generated in **<$TIEFVISION_HOME>/resources/bounding-boxes/TRAIN.txt** and the test set in
**<$TIEFVISION_HOME>/resources/bounding-boxes/TEST.txt**.

The next step is to encode each one of the crops by forwarding them throughout the encoder. On the other hand, the output is statistically normalized (substracted mean and divided by standard deviation).
To encode the crops you'll have to do the following:
```
cd $TIEFVISION_HOME/src/torch/2-encode-bounding-box-training-and-test-images
luajit encode-training-and-test-images.lua
```

The next step is to train the four neural networks needed to detect the bounding box: top, left, bottom and right.
To do it you'll have to do the following:
```
luajit train-regression-bounding-box.lua -index 1
luajit train-regression-bounding-box.lua -index 2
luajit train-regression-bounding-box.lua -index 3
luajit train-regression-bounding-box.lua -index 4
```

Finally, you can test the neural networks results by running:
```
luajit test-regression-bounding-box.lua
```

## Image Classification

Given an image, you'll need to detect where the object is located
in the image. To do that, you'll have to train a classification neural network to detect foreground (your object) and background (everything else but the object).
For example, if you are trying to make image processing on dresses, the foreground
will be the dress and the background the photo studio and the model's extremities
(head, legs, arms).

For this step it's required to have the downloaded images and bounding boxes explained in **Bounding Box Regression** section.

The first task is to generate the train and test data which is something **TiefVision** automatically does by opening the [Image Classification Test and Train dataset endpoint](http://localhost:9000/generate_classification_train_and_test_files).

The next step is to encode all the crops for image classification:
```
cd $TIEFVISION_HOME/src/torch/4-encode-classification-train-and-test-images
luajit encode-training-and-test-images.lua
```

Once all the crops are encoded, you'll need to train a classification nerual network for the foreground and the background classes.
```
luajit train-classification.lua
```

# Image Bounding Box Generation and Encoding (OverFeat)

The next stage is to detect the bounding box for all the images stored in **<$TIEFVISION_HOME>/resources/dresses-db/master** and generate the crops as new JPG files which are stored in **<$TIEFVISION_HOME>/resources/dresses-db/bboxes/1** and its horizontally flipped versions in **<$TIEFVISION_HOME>/resources/dresses-db/bboxes-flipped/1**. The crops are generated with a fixed minimal dimension (width or height) of 224 which is the minimum dimension that the encoder can accept.
For that run:
```
cd $TIEFVISION_HOME/src/torch/7-bboxes-images
luajit bboxes-images.lua
```

The following step is to encode each one of the cropped JPG files passing them throughout the encoder network:
```
cd $TIEFVISION_HOME/src/torch/8-similarity-db-cnn
luajit generate-similarity-db.lua
```

# Unsupervised Similarity Search

At this point it's possible to search images by using the filename from the master image folder **<$TIEFVISION_HOME>/resources/dresses-db/master**:
```
cd $TIEFVISION_HOME/src/torch/9-similarity-searcher-cnn
luajit finder.lua <IMAGE_NAME_OF_AN_IMAGE_IN_$TIEFVISION_HOME/resources/dresses-db/master>
```

To generate a database with all the distances between all the pairs of images in the master folder you should do the following:
```
cd $TIEFVISION_HOME/src/torch/10-similarity-db
luajit similarity-db.lua
```

Once the database is generated, the images in master can be searched using that database using:
```
cd $TIEFVISION_HOME/src/torch/11-similarity-searcher-cnn-d
luajit finder.lua <IMAGE_NAME_OF_AN_IMAGE_IN_$TIEFVISION_HOME/resources/dresses-db/master>
```

# Supervised Image Similarity (Deep Rank)

Deep Rank is a neural network that transforms the feature space of the images into another that is optimal for image similarity.
The first step to train the neural is to generate a database of similar images:
* A reference image **H**
* An image **H+** similar to **H**
* An image **H-** similar to **H** but not as similar as **H+**

Ideally, the training set should be one in which generate a wrong ordering when using an unsupervised approach: **H**x**H+** < **H**x**H-**. The neural network **NN** should be one that generates a correct ordering: **NN**(**H**)x**NN**(**H+**) > **NN**(**H**)x**NN**(**H-**)


The first step is to generate database by using the [Similarity Editor](http://localhost:9000/similarity_editor). The **H** (reference) image is the one in the top-center of the screen ('Image To Search'). Below there is a list of images. The user should first click on the image that will represent **H+** which will be framed in blue. Afterwards the user should click on the image that will be **H-** which will be framed in red.

Once the database is generated the next step is to generate a training and a test set to train the neural network. That is done automatically in TiefVision using the [image similarity train and test generator endpoint](http://localhost:9000/generate_similarity_train_and_test_files).

The next step is to train the neural network using a Hing loss criterion:
```
cd $TIEFVISION_HOME/src/torch/13-deeprank-train
luajit deeprank-train.lua
```

Finally it's necessary to do the same encoding and database generation procedure done for the unsupervised case in order to use it in the search engine.
* Encoding of the images into the new space generated by the trained neural network
```
cd $TIEFVISION_HOME/src/torch/14-deeprank-encoding
luajit deeprank-encoding.lua
```
* Generate database of similarity between each pair of images in the master folder
```
cd $TIEFVISION_HOME/src/torch/15-deeprank-db
luajit deeprank-db.lua
```


