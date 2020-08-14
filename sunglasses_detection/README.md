For the processing of the dataset I was provided with, I have written image_processing.py that crops all the images of the dataset to 24x24x3 images (sunglasses and no_sunglasses).
To do so, I have defined:
* *crop_dataset* that loads the json that contains info (presence of sunglasses, filename, bbox, etc.) of all the faces (in all images), creates a new folder where to save all the cropped images and then, I go through both dataset folder (img & no_sunglasses), ensure that I get non-empty images (not Nonetype) and crop them thanks to the bbox keys corresponding to finally, save them. I print a message every 50 cropped images to be aware of the lasting time.

For the cropped images, I need to know more about their shapes to resize them later and the distribution of labels (sunglasses vs. no sunglasses). 
To do so,  I have defined:
* *get_shapes* extracts all the cropped images and extract each sample's width & height and save them all in widths & heights lists 
* *get_labels_info* that counts the amount of no_sunglasses and sunglasses images in the dataset and then, returns a list of labels list and positive & negative images rate in the dataset.
<br /> Also, I plot the distributions of heights, widths after *get_shapes* (min_width = 23 and min_height = 24) and I plot the ditribution of labels after *get_labels_info* and I get that 98.5% of no_sunglasses (very very imbalanced dataset)

For the tuning of Deep Learning models (convolutional_sunglasses_detection.py), the objective is to tune a selected architecture (fixed type/number of layers) with the following parameters: f (number of conv2D/maxpool2D), d (number of dense layers) & batch size (number of samples that will be propagated through the network). 
To do so, I have defined: 
* *open_resize* to extract an image from a path and adds it to a list,
* *open_flip_resize* to extract an image from a path, flip it horizontally and adds it to a list, 
* *build_folder* that creates a folder with a chosen name & path, 
* *globalize* that make path in relation with an existing directory, 
* *build_dataset* that returns images & labels dataset given an amount of no_sunglasses images (which are a majority in the initial dataset), 
* *separate_test_train* that returns training & testing sets from the dataset and TEST_PROPORTION, 
* *labelize* that changes a vector softmax output to a categorical vector (from [0.3, 0.7] to [0, 1]), 
* *plot_epoch* that takes history of model fitting and some metrics and that saves plots of each metric in a chosen filename, 
* *multiple_append* that append multiple elements to multiple lists and finally, 
* *architecture* to build keras model thanks to number of dense & convolutional layers and their parameters (number of filters and number of nodes),
* *build_model* that takes no argument and do:
  * create folder for callbacks & plots savings
  * create scores, col_f, col_d, col_batch, conv, dense & execution_time containing: f1-score, f value, d value, batch size, number of Conv2D, number of Dense & prediction time
  * train a chosen architecture for each (f,d,batch size) with a 5-fold cross validation, then:
    * compute average f1-score and its standard deviation over the 5 tests for each model
    * saves an excel tab with columns containing network parameters and his performances (average f1-score, f1 std)

For the tuning of best models from the training of convolutional_sunglasses_detection.py (anti_overfit_5_best.py), the objective is to tune the 5 best models from the excel before (fixed type/number of layers) with the following parameters: known triplets :(f, d, batch_size), and differents f_rate (dropout rate for conv layers) and d_rate (dropout rate for dense layers). 
To do so, I have defined: 
* *open_resize* to extract an image from a path and adds it to a list,
* *open_flip_resize* to extract an image from a path, flip it horizontally and adds it to a list, 
* *open_blurr_resize* to extract an image from a path, blurrs it and adds it to a list, 
* *build_folder* that creates a folder with a chosen name & path, 
* *globalize* that make path in relation with an existing directory, 
* *build_dataset* that returns images & labels dataset given an amount of no_sunglasses images (which are a majority in the initial dataset), 
* *separate_test_train* that returns training & testing sets from the dataset and TEST_PROPORTION, 
* *labelize* that changes a vector softmax output to a categorical vector (from [0.3, 0.7] to [0, 1]), 
* *avg* that computes the mean of a list, 
* *save_excel* to save an excel with a name depending on the anti overfitting technique.s used, 
* *multiple_append* that append multiple elements to multiple lists and finally, 
* *get_flops* to compute floating-point operations per second for a model
* *architecture* to build keras model thanks to number of dense & convolutional layers and their parameters (number of filters and number of nodes)
* *build_model* that takes no argument and do:
  * create folder for callbacks & plots savings
  * create scores, col_f, col_d, col_batch, conv, dense & execution_time containing: f1-score, f value, d value, batch size, number of Conv2D, number of Dense & prediction time
  * train a chosen architecture for each (f,d,batch size), then:
    * compute average f1-score and its standard deviation over the 5 tests for each model
    * saves an excel tab with columns containing network parameters and his performances (average f1-score, f1 std)

To check the performance of models (+anti overfitting technique.s) based on the excel produced by the last codes. 
To do so, I have defined: 
* *open_resize* to extract an image from a path and adds it to a list, 
* *build_folder* that creates a folder with a chosen name & path, 
* *globalize* that make path in relation with an existing directory, 
* *build_dataset* that retruns images & labels dataset given an amount of no_sunglasses images (which are a majority in the initial dataset), 
* *separate_test_train* that returns training & testing sets from the dataset and TEST_PROPORTION, 
* *separate_test_train* that returns training & testing sets from the dataset and TEST_PROPORTION, 
* *get_flops* that returns the amount of floating point operations for a model in h5 format,
* *separate_test_train* that returns training & testing sets from the dataset and TEST_PROPORTION, 
* *analysis* that takes excel file of F1 score of each trained model, prints the average score over all the models, the 20, 10 and 5 best model and its standard deviation ans highlights in which case of anti (or not) overfitting techniques we are (batchnorm, etc.),
* *plot_confusion_matrix* that saves & shows a confusion matrix with a Blue Heatmap,
* *architecture* to build keras model thanks to number of dense & convolutional layers and their parameters (number of filters and number of nodes)
* *model_analysis* that takes excel file of F1 score of each trained model for an architecture finds best model for a specified architecture and finally, saves & show a confusion matrix for this model with its flop as a title.
