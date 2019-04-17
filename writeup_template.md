## **Behavioral Cloning** 

---

### 1.Steps

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### 2.Model Description

The model archetecture referenced the pilotNet of NVIDIA, which is shown below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Cropping2D     	    | cropping=((50, 20), (0, 0))	                |
| Resize                | outputs 66x200x3 RGB image                    |
| Normalization         | net = net/127.5 - 1.                          |
| Convolution 5x5		| 2x2 stride, valid padding, outputs 31x98x24	|
| RELU	      	        | 				                                |
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 14x47x36   |
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 5x22x48    |
| RELU  	      	    |  			                                    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x20x64    |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 1x18x64    |
| RELU	      	        |              			                        |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 4x4x32      |
| RELU	      	        |              			                        |
| Fully connected		| outputs 1164 classes                          |
| RELU	      	        |              			                        |
| Dropout	      	    | Dropout 0.5       			                |
| Fully connected       | outputs 100 classes                           |
| RELU	      	        |              			                        |
| Dropout	      	    | Dropout 0.5       			                |
| Fully connected       | outputs 10 classes                            |
| RELU	      	        |              			                        |
| Dropout	      	    | Dropout 0.5       			                |
| Fully connected       | outputs 1 classes                             |
| Tanh	      	        |              			                        |

### 3.Training

In training process, Data augmentation is applied using `np.fliplr()`. Adam optimizer was chosen because of its faster convergence and reduced oscillation. Batch size is 32 and total epochs is 50. Earlystop is applied with `val_loss` as monitor in order to get the model that could be generalized to the validation data set.

The loss curves for training and validation data set are shown as following:



### 4.Results

The model training and validation loss curve is shown below:


The testing result is shown below:



### 5.Possible Improvements

