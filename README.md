# Assignment4

1) how many layers 
> number of layers depends on the input data 
> hardware 
> accuracy to achiecve

-------------------------------------------------------------------------------------------------------------------------------

2) 3x3
> kernal is the 3x3 matrix which will extractor of features of the image.
> with the  even number of pixels in the computation it is impossible to locate the result in the center of the cells used to produce it
> 3x3 kernal will extract the features with minimum loss.

-------------------------------------------------------------------------------------------------------------------------------

3) receptive field 
> receptive field is defined as the region in the input space that a particular CNN's feature is looking at.

-------------------------------------------------------------------------------------------------------------------------------

4) maxpooling 
> it is the computational layer in the cnn which will reduce the channal size so computatational over head is reduced but the 
reduction in channal size will cause the loss in the information only outputting the important information. 

-------------------------------------------------------------------------------------------------------------------------------

5) position of the max pooling 
> it is placed after the convolution where to reduce the no of channals and it is restricted to 3 layers before the final convolution layer

-------------------------------------------------------------------------------------------------------------------------------

6) 1x1 
> it will  multiply the each channal with number.
> lesser computation is required for reducing the number of channals 
> less number of parameters.
> use of existing channals to create complex channals.

-------------------------------------------------------------------------------------------------------------------------------
 
7) concept of transition layer 
> max pooling , batch normalization and drop out from the transition layers.
 
-------------------------------------------------------------------------------------------------------------------------------

8) position of transition layer 
> it is placed after the convolution where to reduce the no of channals and other parameters and it is avoided to 3 layers before the final convolution layer.

-------------------------------------------------------------------------------------------------------------------------------

9) the distance of maxpooling from prediction 
> 2 or 3 layers beyond the prediction layer

-------------------------------------------------------------------------------------------------------------------------------

10) kernals and how do we decide the number of kernals 
> kernals are feature extractor. it will decided by the network designer based on the input image size and the hardware.


-------------------------------------------------------------------------------------------------------------------------------

11) softmax
> the softmax function is often used in the final layer of a neural network -based classifier

-------------------------------------------------------------------------------------------------------------------------------

12) image normalization 
> it is the preprossing on the image data to bring the data in to 0 to 1. which will helps to avoid  the large computations

-------------------------------------------------------------------------------------------------------------------------------

13) batch normalization 
> batch normalization will solve the overfitting problem .
> batch normalization should be added before the prediction layer

-------------------------------------------------------------------------------------------------------------------------------

14) the distance of batch normalization  from prediction
> one step beyond the predication layer

-------------------------------------------------------------------------------------------------------------------------------

15) batch size  and effects of batch size
> Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration. 
> depending on GPU bigger batch size might speed up epochs
> larger the batch size lesser the time for each epoch
> ideally batch size should be more than number of classes
> bigger batch size is generally better for better learning

-------------------------------------------------------------------------------------------------------------------------------

16) number of epochs and when to increse them 
> number times that the learning algorithm will work through the entire training dataset. it will incresed when the modal is under fitting.

-------------------------------------------------------------------------------------------------------------------------------

17) drop out 
> Dropout is a regularization technique for neural network models
> Dropout is a technique where randomly selected neurons are ignored during training. 
> reduces the gap between testacc and train acc

-------------------------------------------------------------------------------------------------------------------------------

18) when do we introduce the dropout, or when do we know we have some overfitting



-------------------------------------------------------------------------------------------------------------------------------

19) when to add validation checks
> always 

-------------------------------------------------------------------------------------------------------------------------------

20)how do we know our network is not going well , comparatively , very early
> By noticing the first 2 epoch in the network

-------------------------------------------------------------------------------------------------------------------------------

21) when do we stop convolutions and go ahead with a larger kernal or some other alternative 
> before the prediction layer

-------------------------------------------------------------------------------------------------------------------------------

22) adam vs sgd 
> sgd 

  *it will not perform the computation on whole data set which is redundant and inefficient — SGD only computes on a small subset or random selection of data examples. 
  *SGD produces the same performance as regular gradient descent when the learning rate is low.

> Adam
 *Adam is an algorithm for gradient-based optimization of stochastic objective functions.
 *It combines the advantages of two SGD extensions — Root Mean Square Propagation (RMSProp) and Adaptive Gradient Algorit      (AdaGrad) — and computes individual adaptive learning rates for different parameters.
 *computes adaptive learning rates for each parameter.

-------------------------------------------------------------------------------------------------------------------------------

23) learning rate 
> The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

-------------------------------------------------------------------------------------------------------------------------------

24) LR schedule and the concept behind it
> learning rate at the intial epochs needs to be more once the training is about to complete it has to less this is handeled by the LR schedule 
> adjust the learning rate used by the optimization algorithm.
> A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas a learning rate that is too small can cause the process to get stuck.
