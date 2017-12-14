## Project: Follow Me
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

[//]: # (Image References)

[fcn]: ./fcn.png
[my_model]: ./my_model.png
[encoder_decoder]: ./encoder_decoder.png
[learning_curve]: ./learning_curve_epoch40.png
[final_score]: ./final_score.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

You're reading it!

#### 2. The write-up conveys the an understanding of the network architecture.

![alt text][fcn]

Before training process I gathered more than 7,000 images using QuadSim.

I was inspired by the diagram of the FCN Lecture. I stacked 3 layered encoder and decoder. Higher layers have bigger size of filters than lower layers. Filter size of each encoder layer is twice of the the previous one. Vice versa in decoder layers. After several experiments I found that more than 3 layers can make the fcn model overfitted.

![alt text][my_model]

```
def fcn_model(inputs, num_classes):

    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc0 = encoder_block(inputs, filters=32, strides=2)

    enc1 = encoder_block(enc0, filters=64, strides=2)

    enc2 = encoder_block(enc1, filters=128, strides=2)

    #enc3 = encoder_block(enc2, filters=128, strides=2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    outs = conv2d_batchnorm(enc2, filters=128, kernel_size=1, strides=1)

    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec0 = decoder_block(outs, enc1, filters=128)

    #dec1 = decoder_block(dec0, enc1, filters=128)

    dec2 = decoder_block(dec0, enc0, filters=64)

    dec3 = decoder_block(dec2, inputs, filters=32)

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(dec3)
```
#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

I set parameters as follows.

```
learning_rate = 0.01
batch_size = 32
num_epochs = 40
steps_per_epoch = 50
```

After several tries, I set `batch_size=32` just for the my pc gpu memory specification. I know that `batch_size=32` is reasonable for many deep learning samples.
learning_rate 0.01 is also typical setup for the simple deep learning samples.
I decreased steps_per_epoch to 50 because I want to check the learning curves more frequently. (The original setup was 200.)
`batch_size x steps_per_epoch x num_epochs/num_train_samples` (32x50x40/7,000=64,000/7,000) means that my fcn model trains each sample more than 9 times.
This setup is enough for the small simulated test.


#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.
* The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.
  * I used separable convolution layer and  batch normalization for encoder blocks. By using separable convolution the number of my model's parameter is reduced. By using 1x1 convolution in the Separable Convolution, we can implement fully connection between layers on small sliding windows (receptive field). These are used for the Semantic Segmentation or Object Detection.

* The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.
  * Fullly connected layers are used as the last layer of Recognition and Classification task that do not need spacial information. It merge all information to nodes of the next layer from the prior layer.

Additionally Bilinear Upsampling and Skip architecture and concatenating technique are used for the decoder blocks. By Bilinear Upsampling, encoded information for the segmentation are expanded to the pixelwise final result.

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

* The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

![alt text][encoder_decoder]

* After passing data images through the encoder blocks, responses of neurons of each layre are merged into smaller regions.
  * Passed tensors widthxheight dimension are shrunk. But size of channels are increased. Spacial information are compressed.
* In the decoder blocks these values are expanded to the same dimension as input images, after passing decoder blocks.
  * Encoded Spacial informations are reconstructed. The size of channels are shrunk.
  * To compansate detailed shape information in decoder  blocks skip architecture are used.
* Since there are only convolutional network without any fully connected network in encoder and decoder blocks, all layers preserve spacial meanings.
* Because of these key characteristics fcn models make the final result as segmentation labels by inducing appropriate loss setup.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

Traditionally walking persons are relatively simple objects to classify. In this simulated situation, all other background objects are plain and simple so simple fcn can segment the target 'hero' very clearly. But If the target changes to the animal with 4 legs, the classification becomes Harder. Shapes of 4-legs-animals are more diverse comparing to the shape of walking person. It means that we need more data and complicated FCN (deeper FCN Model).

The Original FCN Paper and other related papares deals with various labels including animals, cars, plants and sofa. If we use a model with appropriate complexity, we can segment all types of objects.

### Model

#### 1. The model is submitted in the correct format.

Final Result weights are here:
* [config_model_weights](./data/weights/config_model_weights)
* [model_weights](./data/weights/model_weights)

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.

My final result is more than 43%.

![alt text][learning_curve]

![alt text][final_score]

