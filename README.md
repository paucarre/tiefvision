# TiefVision

End-to-end deep learning image-similarity search engine.

**TiefVision** is an integrated end-to-end image-based search engine based on deep learning. It covers **image classification**, **image location** ( [OverFeat](http://arxiv.org/pdf/1312.6229v4.pdf) ) and **image similarity** ( [Deep Ranking](http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf)).

**TiefVision** is implemented in [Torch](http://torch.ch/) and [Play Framework](https://www.playframework.com/) (Scala version). It currently only supports Linux with CUDA-enabled GPU.

# Results

Here are some of the (best) results for a dataset of dresses:

![Flowers](https://lh3.googleusercontent.com/qHnAydafymTdspCFkQEutnVjDVBUhqeoqybH1_26KULE0nXQsOvyuELc2kj53PLB5gfEEM_tjenJ7TLAW5EFW39mIBmJgDJ0M6wpSaAwFwzAWHme-Y9ROlBES5S0s-5wQJrnEST_TzR7A55fX7URt0k8xH5GLDjEzGSaY7SCE96vSAveS_XTfp8FRxDa8HhdmSV1BZS5HmUawY3bhmHi9UyXY2XjLabXOPpvmUlgXsH4ee-1DoSBpugdoxXrct63gQVZVsDy07ikpUrwqM-JanSULL4FybjYIXgUqIEqUek6hrGGSHAKwD5xqhVtzOCQFnvtpTLya_2lAUBsZ5sssHYmaAJs-_NGiXasSpw4oy5hQi4Q50kqxdKgEaCtFizZgyq-7dK7UbeuXoQbvP43iF9hhA4_U16y03F04lMOIdlW5wN83PKQQ6TDHhgLaODYsOJ4ulla3ftYttgJ2WO6tpX7eUwR5hRwajB1s9kWN0ta1JA1JXO6ECFwReDjoqNa9LiG-MtrNoK_UzFKUqi620FkQ_y6KpblD8NvPiqtfoRKD8phMJMarEpjJS9XEqY9-uOd=w1788-h995-no)

![Gown](https://lh3.googleusercontent.com/r79vpytThecVvh8bCcRa20H3kqyACe_xAocu7CHsv4jpyH1pPbqpTzjhCQmtfvrsuqb9wnlqY6MtfJH7L7e2yQBcACD_LmgAYipVqv2CeUxnTvOIa9XsvvSKrwD6X07juIejfD7TC6PtXb47Xudyw0ZhG8-M0MCqO-Ues9ElAceZyKlVX_-ScImDxhh7a1s5UlsNFV2yiOekV9JLxgB2a0VxHQGoLg5IxVXDs4NsyWiNJWasLLbLYUjAIrVgo-n0MeikSwWTFCAPr9m5497BNWzzRDtc0s-hPm8-S7XoJQhkvIbYyk3u_x284oFrE9N_Uxeg5CSZOE6sEQPBRZhLzOPcJXZaoIHIx47VQCmnGqVTxbWM17Ci-_y9pXpqJ1lXKGos0pBuvqisnlcQiZyyKvL1lXjqT1B9Jy_FISCmjFlaSS21QqpwUx2rwvxfUv3olJLDE5SZZid_vhYr9F9z7b7Wiu2t8jhgyLnoOgNwULw0jf47uDM_PmSvQXcDK8KTy2MJ8CxOJ1bhfiF_YM6LVHZCHrCgF-0KwCieDoPctxS-2HPTLjxtg99-Id9uwF9R8CIK=w1798-h995-no)

![Long](https://lh3.googleusercontent.com/MRGJeRWwyf8YXrnr4YLdJLS8X11VFAskS7K23OBwzF7PxqZCQcFPJaBY97b6O9HqN569FKLcANTlaJFPkAwcXKxtOeH0nXGOrfR70baCGOGAjowSR_-x6a7ZFgfaSGSzKEG6OodX3zrH1Cjgrs2iAk1EJmv1QXe9wdrftMsN45K6DweIerN6RupMGxIXeEwr8mFyb9ZEfvjcnWdgTQ-uWV1Nn3OwV6UdHH0nxzyG5Q0-NW37kJV8LXgwV_zQmqlUFOf5gpa0NckdO0kWnY589g1X8A7FUcpWhRcgMpBhf3sjla_5GeBVUJjVM4tHnymjIE65H-B45ptFbGx0B0AWbI-9yT_-wcHoaQKkg2lZjw8pk1IJ2l7RCOWuuzJphepdgtX4Wr4oR-unY5WB8VvMlX0sayQBwyCGu709R-3zp7TPv3yrG09RTdGkev5hqxu4Gcolt6kyAcIK5cKjMlERAvcNm8ILEJSZDzVXOOhT7GQmNhH3EOk1WZcTmcNVSLr06HaJFVHenhfSld84Wa-s_a_xf2Z_m_t7gt0EMK6kgU-WCIyDD07kts5K1RPT874VLJn5=w1790-h995-no)

![Neck](https://lh3.googleusercontent.com/YzFR61IKk57KwvnASs5KUeeHOaGK_emoVBn09ahf0JqISN7oPb_KwYBsIIjToI04vGbLhe_676b67KqVDRAUS3-b5aZt39n4jTgdMYjuifhiB8fTuhDn_uewH8irGrLxNSfe-i__9Oqp-IP227Qq5xN94gOTutPrLcL2QiP4MOgqCU6oBZQv_ZHLm7bqBEaMNCU3Iyz4xkLEF1BoxWQ55AEZAkPKPmVP3LVMIYDtdhEBts7QQ6QrmBrFF-5ZjSNDx6wTwIXg_3xiennh471u2uuDzLyIHFhbjF_yFQO5rfXb2Zv5GHbgqd0Ehy7Agvlh5gs0FPa1CjTJaRcBPTFXghNpcfBsQmyfcHBMM8wY7xFGohkqyUI-iWzatHv1oEWBShzkbNWpci9xOEzx2uXAZYobFgA_ofJZjB8eAc64CR9Vi86syvSJYPVhQFNrE_BXiPf24PAq76vTItVsv1fSr12eT2IxLBMDKMM3RA8jUddCDwzS7fGz6_HaDyw2yICpZVdaVpiQTEWfwVUva_Rno8g_yi3PgNCnN8wO0yjFc9bu4dDMrUhtCtnEbf7DPM7sCzR2=w1796-h995-no)

![Black Party](https://lh3.googleusercontent.com/eNkvxzMgq3pw1R5JkI7Uu4MW1oOchVU3YQlA2i9VqnbNXwCosCceQn2nxr_FkPgB-11QAuR-2P9KyYaWkoXB0ugmRHCQFhuEb0wMTEikh1_ULgI2gpQLDogs8jVlE8lDxuuVHTohzmjAetdjNY5lpkkBhXx4RjvVEI4X8Br0kfdn4kmhoowhpljFSRQB71KUoIdtfDTxltCsadGNOX6h4gxFlJzxb7C13ZhuXDiBQpZjHWMhPzBs29hq2qMbnEGsyUCwd-tDKMYTirAe8Vg2bJjtVg95fywaSwwT0ZWjSR19x17VkG_xvGJKBsSnocZj0bEwjz0DUmaUbqQWGlp-VMBN2gH2JYN6HA0bGz_CEBcbAn1_C8PaBYeEIGiWkclNQzQm2Do0TMNJ0T6nVcvXssBghvoypJl0iE0qYf2v2tiaFTaLqS10H5hXYju9P9Si7YdJdiPMfYCCmoCRLFQzsV4GVfEoFpoDJG4LT9eT_ZJM0JEuEdVt9yphoMdhxtGqex__Alc99ojBlPfFoTWfBU_hPZ2lyUs4CPZkGHhHkmu_VrphcKXkjCS-NuIrx7AeZkhD=w1851-h990-no)

![Patterns](https://lh3.googleusercontent.com/SbvE1FhJr9o0gbRwHsnK02gCdMOw8KynPppZKGM4mEobtLQkwTaEz9QIpf_Ym5KO4Up_VshxaLdQ5MRmUjecATLsTQSQtXHVfKiMe0vzJZ2lqvblwRkO1AcsfBuWQRfP16W74VQlHMiEl2vl5YPwrd8LoGq16N659IwH58xTio8ex5TEU4YB9WcDIxZfYrOrGLsvdHFDjOEUyohfOHpCecmQ8oBX7MnU4yQeV_7PsrS_bOrYMtroxzXFlnBydu1ZXTbyg0LdsEDHJgIQX1oYpSnS49PW6JP8anDKzLSNlZ0EkmYuLPGjH0hVSJCHs5YJVMwehT_vtY-WgA7QrAK2X2ZvH0vGFmuyndRN_xGxErQI1foAzYYeotB5whzTxOcUVVCsJtdVx_uoHVRahIiLCcUi54ubJZazToleu7Mm4-lMKzUH3QUWRIsgN8ZbpWq88xo0k_4kyLFKZ5WoIsa3U1seK9Ubkf8n5YOtUV1lHbNypVgljUFG3pYqwcN5Mh2IHZDwvoGA4Te_aGhgu3BWvMS6eAsj-wkxca9uHd0s6HKsMS6ViDGPm-O4JSS784zSDS9-=w1815-h995-no)

![Squares](https://lh3.googleusercontent.com/wfTvPzHu4SFQiy-ACNFsKlq6EGFeABMbVmR4W0QsR2kVYd2RUS9JNyza8Hz7yPAbxje0Klj9It_S-WZGF1G7PQL9q8KAszqIwE3vYl_aIDwCut1UKQI-jSuAiHreEyNYCXhZrgcD9fA3cNMEt5Xyg8azZQ5LvuRHf0VrzK5KZ7VfSQseXIuZx8f8CxUf0W4xl0TTykkhpzP7Kv-sMlE5zoq6OUaC7ym-ok20n2pNPLB1ERWVXI5ecZUWEODbQdUUoueM5slTGBA0uJrwpxKeA2hWpogqtxSP5dySuIjh2YLH3pppFUpX8a3OuVumo2-SAseOuKx7eB9j7ekr8R6z5NGc9DYx5jWg5OGOLYSWWrbM3l2f0X37Z6WtDO13aVoR6YDOXUTZHo0M36ISfCm5JIuqZ6GB8_1U6T9ltnKVpTxkSr8e3FwIQO6dqV7t0v3-prg-rfuK1kRagUMtOERA6iqLvUCjvdO64TUfe1BL39-QlDkqr1qe6I0AmtKUQHqxgBfVieSci3oQbwEGK2PA3nheTUQrQNnF_N0elYGN94O_FWThAjFwPOPvQTuUshL7Yj7O=w1755-h995-no)

![Stripes](https://lh3.googleusercontent.com/Ue8HDZOxCL9bwBozV54WsXWI7kIy75o30blZIEf-bjzVUUtSXdjzR15zoNjgEK8A5VTa8ct5Z6_oXIphCgwWz30VX_98VDCUyIDKRfDiJzcvJpUvnE1szAPBpbOjvALZ3VbO9-BIc-PsRGGkAdfxegTjX2pgzyNsKx7wkjjDaTcWy-CLQrC_C573srngxdVJoYuPdBCGFXGDiqfRHkhDiDceOgMsyTD4nV5iAaZTJ27yLHVXUp4lkuEVn3tTxDtmSIM28bJreMW7VLM6eTpdHfcb2j-GC494DsBFJxMHkNp89lESRoo1OfqFkEgBWuVrhTViB75SwZWLtpQ6HS8PhayYrSMhnTKuHbeUtAVJPUuE3WKuGSC-_3DoxEVP9eW6vFoaWMVi6yoZegOOs7VUMBs0QifvSPz3VlAJ5ormA9ApJskvoXqBHiFUdXL88ogYgKG6RD_9dNCtRMgdkbGm0qxCAui7w46kb3KfWNoFFDtWCQ5Kl1yZkCLtTWew6hBlkOcdPsdXpAtb9AzNmUISDqqJSQyaJtmwy-nugJbcxeariVgFN3eVLiqeW4knxKDeh0Nz=w1748-h995-no)

# Features

The project is divided into two module groups: **Deep Learning Modules** and **Tooling Modules**

## Deep Learning Modules

The deep learning modules included in **TiefVision** are the following:

### Transfer Learning
**TiefVision** transfers a simplified (without grouping) **AlexNet** network that is used for encoding purposes.
The steps involved in the transfer learning phase are the following:
* It splits an already trained **AlexNet** neural network without grouping into two neural networks:
 * The lower convolutional part that acts as an encoder of high-level features (“image encoder")
 * The upper/top fully connected part that is discarded as it’s meant to classify images for other purposes (ImageNet classification).
* It reduces the last max pooling step size from the encoder neural network (lower-part) to increase the spatial accuracy.

### Image Classification
The image classification module performs the following steps:
* It encodes all the crops from the target image (e.g. dresses) and its background using the encoder neural network:
 * **Target Image Crops**: crops of the images in such a way at least 50% of the crop is inside the target image bounding box. For a dataset of dresses, at least 50% of the crop contains a dress (it can include up to 50% of the background).
 * **Background Image Crops**: crops of the images in such a way at least 50% of the crop contains the target image  background. For the example of dresses, at least 50% of the crop contains background.

* It trains a fully connected neural network to classify the target image crops (e.g. dresses) and its background crops (e.g. photo studio background).

### Image Location (based on [OverFeat](http://arxiv.org/pdf/1312.6229v4.pdf))
The image location module perform the following steps:
* It encodes the **Target Image Crops** dataset together with its normalized bounding box delta (distance between the bounding box upper-left point and the bounding box coordinates).
* It trains four fully connected neural networks to predict the two relative bounding box points:
 * Two neural networks for the two dimensions of the upper-left point.
 * Two neural networks for the two dimensions of the lower-right point.
* It extracts the bounding box filtering out background crops using the image classification neural
network and averaging the bounding boxes using the bounding box neural network.

### Image Similarity ( based on [Deep Ranking](http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf) )
The similarity is based on the distance between two image encodings.
**TiefVision** trains a neural network to map encoded images into a space in which the dot product acts as a similarity distance between images. As the encodings are normalized, the dot product computes the cosine of the angle between the encodings.

Given the following triplets of images:
* **H**: a reference image
* **H+**: an image similar to the reference image (**H**).
* **H-**: another image that is similar to **H** but not as similar as **H+**.

It trains a neural network to make **H+** closer to **H** than **H-** using the Hinge loss: **l(H, H+, H-) = max(0, margin + D(H, H+) - D(H, H-))** where **D** is the dot product of the two images mapped into the neural network’s output space: **D(H1, H2) = NN(H1) · NN(H2)**


## Tooling Modules

**TiefVision** includes a set of web tools to ease the generation of datasets and thus increase productivity.

The current tools are the following:
* Visual Bounding Box Database Editor

![Bounding Box Editor](https://lh3.googleusercontent.com/FgNHIA_p7YR-sIXkgyt_FEhxjE1vNdPAHfdcGcXlSNvMLCcqz3dmgJdyxw4ejpPcJTEPLzXwfkWWPHCj6RYlfa9T3bwKrtwb7-EV9y8J-PG--AKCXXbmgBisfJ7DlN-v8dH7nXa7oOgIMAxD2uqFCTAjsLSrt8U2ZJO1g5lu_FrFKY9D8FPMGneYcQhv3W55bULAteObgc7h-noaFn2pBkD4V42vgL_cUHwZqYvqLyaKXvJEsvv8B7qgqyJI3tjgHmPzPvozP1teFstUKBxVWZTweD-AwAqLpJ_VAZu2YuVDCKK5I1hb7yhPLIKNoTWPQWACmIPrayVTI4ZF04XkBl0BZuakegcphis9GXZyle6tfHgW9n0GTOM2FIJb9rMrc8hV3wizVxIiQghu2PYwT5nGOEG4VBL3dvTIheXsByg58V3lPrQZdqy0KAZEgAMRsxDN-GsLi48M1A57p7GZT6bSbOJHDzCiUGrHPPXognEfk6Gr7g5H2nS7RKERmjDeJS-MlbZnK4n6NUoYhE3ClV7eK6p6onTtRDRIJXoZqBQSPP5U6-A5JyPL5KrQ5sqUTYcy=w493-h873-no)

* Visual Similarity Database Editor

![Similarity Editor](https://lh3.googleusercontent.com/sU5-eCYORgcuJKMQd8EZHkHg9MF37s2EyV3Awj_CdTkPScWdANOOxE96kK58aNG3Y43u8dVo6weL5uPwjPc8I-IlZVSL7Z93TOycN3X6iKlLP1Ppo0PRo2qNUmLZ_VrZmmEj4jn4qaqAU7-knyTXmWXV-I80oghscP1FN31TwinXvIhq0qBylZMLnOM9Nrdvy0NpOx6v3lcMjcN5jTOV1Atm_ClSLUU2onUVXe7qp81h6gRW_gUzfsC3bwHi4Ea3TzGo68wm-RC4bK9x_DH0G7vM7x7f7fKAv8Dw3Lx5yZG-NWMdkb-JRT2r8EcvPQ6OFFe3uluh4KO-aUVmevUcCZrBzVROpr_Ujy-r_PGkncnYzobGuws43X80ZeBbODGGhbhR92p77KUOxn3sI5NxXQnI0rutJpL2KW7712GBXQJMl0YSbl_JnCdeEaa46GlegEhdacw4DbW3QtX9o_AbWSX8dsnA58CbRW5rU49K2LEj6n_mU-5ez7VuSsa7A6tp1nn2LL0w-LmZeei1KjVQ61B4zr9jw-_nlGJhrJAOUHrfICuqerU7HcSzQnO_YV5G9oVW=w1056-h995-no)

* Web File-based Image Search

![File Search](https://lh3.googleusercontent.com/2OmpaVRHg9crrWvSfL58NZzp_svwI_GZOt22eDuI8FIdP-o0kPz-kIllrlmwBm4z51_SEJUZc3ZMkfKk6Y2QAsE3n6XU90asuuEkqM-eA54OowGoILbjD_ADQ-nG1avyq8vmFYgYldXW3QC1Wdy7FKG3cuDiPd7zEMjZ-Pac5zIDBAnXE-dryiSShlaqcokdILnMsVRVOtIDb2IYZi-3-6WPpYO2UKLZ0FqahRVCXPF6-W-auUChT2pXSGRVFkfCABT0MSpOZbm_FpbxTcdRkpRNS8_9FHUNrTwNoRBdw0hWnDVj8-tJ9PjiKdV6lxWJQ9Oe1-8QyvgBBm9MkCyh_mha83gWQuQJCtxSjo4rzZSNreIwF8_HJtRdiCqWTJWl2ImJ91EhBWYkivgjvqA4uh7eFxUPpwUImKch4COlljfXbSgKWecece6WHhdpDkL5rHOEKSlY4ob-DEufaEI9CFB_zUL8VwuJrztTp4NFmV5a0A8s8MJfvXdUkcP-k-KMu0CodFPHNdtbNorvNyhWg_yiWjQ1BMaODvSa1vyPGPsqhkcg4eDbCietqTMGrO67EWpt=w1543-h377-no)

* Image Gallery Browser (search upon click)

![Image Gallery Browser](https://lh3.googleusercontent.com/f0fa4rDOq0iA_E_8zW3M90d-QomvGyV2cbRR3zv0Un4sKODedHPVdnh1zdaQH6Z-lGGHhLoKf--JOvvjh2xKoBLNOUQNF1nQ22Ex9rsMT6ZI_BKeILrqICdU95_QncEIGrpg0ahZfU4mC0GECAPNPJ2PFzRak3GKFEOLK-v8eu0Xuk592fZ-ucdc33ZrAr-1VuFuiyST_m8Q14dOViD19iBDgTx_qtkG7s_ElIFTyXn7q2PULKCNKn1FvOlp4k2r-riBvfSSDb9-H29eiy1rw9KH3ZcOoFBHQq2s0RAKu3dgzY6P3PltRZFnDuyFz2nQ_G7cVCYwRltsa5QqKKc76OEiOm9OzHvtX2wjVwnSZQml50BWZLXWdbY-ZcZEFIeRQWh-kjGcwQM5t-K0JRSO6YLsjUI0TbC9j6rMTxNOc1sJAiET2Fb8Z1CefaScKW-wesWHid3nGzjGgesvurFQOOJLgoIVPjzRfm9CkmTWN-PoWbz0wnPvZsYKuR1yaE_5IJJ-VkueTjyxV7JdyWc_cB6nrsPgA99NCJ4Fg4tbqYF8PKj01txBQ4UmCN8qgg8aD7ey=w1751-h995-no)

* Automated Train and Test dataset generation for:
 * Image (crop) classification
 * Bounding box regression
 * Image Similarity ([Deep Ranking](http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf))


## User Manual

There is still a lot of work to do in order to set it up for a generic project that is
unrelated to the dataset it is used for (dress style classification).
Furthermore, there is still a lot of work to do in terms of documentation and also 
in terms of ease of use.
Nevertheless, bit a bit the project should be improved to make it simple and easy to use
for any dataset.


### Requisites
The current mandatory requirements to make TiefVision work are the following:

* Development machine with nVidia CUDA graphics card
Note that there so far no will to remove this requirement. I might move it to OpenCL
at the time Torch fully supports it for all neural network layers and it's mature enough
(e.g. other people also use it and it doesn't crash all the time)
* Linux OS ( Mac *should* also work)
* Latest version of Torch 
* Java Development Kit 8 (copyrighted by Oracle)


## Image Classification
TODO



## Copyright
Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
You may use, distribute and modify this code under the
terms of the [GPL v2 license](http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).
