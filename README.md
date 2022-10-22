# ImageClassifier using Convolutional Neural Network.
# Google Colab is used for computing.
# Execute all the steps in order to avoid errors of importing packages or undefined data.
Implementation :

Fetching Dataset from Kaggle

Firstly fetching the dataset from kaggle is done providing the kaggle username, key and the API link copied from pertaining dataset page in kaggle.
The downloaded zip file is then unzipped to obtain the directory with three class subdirectories.

Pre-Processing of Data

Now the pre-processing of data is done firstly by verifying if all the images are of expected format. Here 'imghdr' utility is used to fetch the format of the image and OpenCV is used for working with image path. If any images aren't of format jpeg, jpg and png, they are deleted.
Secondly, splitting of the data provided in the dataset is carried out. For this a directory named 'splitdir' is created followed by 3 sub directories of 'train','validation' and 'test'.
All the images of dataset are shuffled and 60% images are split into testing, 20% into validation and 20% into testing.
Images are scaled so that the pixel representation would be between 0 and 1 by dividing each entity of pixel representation by 255 using Image Data Generator. This way train_generator, validation_generator and test_generator are created.

Build, Train the Model

Tensorflow provides two model building API's namely Sequential and Functional. At a high level Sequential model is used when one input pertains to single output and Functional model is used when multiple inputs pertain to multiple outputs.
Conv2D layer performs the convolution with the provided layer. For example Conv2D(16,(3,3),1) 16 indicates number of filters, 3*3 indicates pixel size of each filter and 1 indicates the stride move of one pixel each time. Activation 'relu' is used here, which means output of Conv2D layer is passed to relu function to make the negative values zero and to retain positive values. This way output is modified.
MaxPooling reduces the image data considering 2*2 pixel by default at a time.
After flattening and dense function application to further condense the image data, three values are obtained. In order to make them the probabilities, softmax function is applied.
While compiling the model, Adam optimizer is used, Categorical Cross Entropy loss is used because this is multi class data where if it is only two classes existing we can use Binary Cross Entropy.
Model summary is printed inorder to check the condensing of image and the size of output obtained along with total number of parameters.
Inorder to logout the model training, logdirectory is created.
Model utility provides two functionalities, model.fit for training and model.predict for prediction. 
Initially, model is trained and validated against the validation split images. Here, epoch represents the duration of training. Both the training, validation accuracies and losses are recored.
Using the history, graphs are plotted between the training, validation accuracies, losses and epochs.
Also the graphs for all the experiments carried out with the hyper parameters are represented.


Test the Model :

This indicates the model.predict functionality. For the test split images, classes are predicted and printed in the output.
