For the processing of the dataset I was provided with, I have written image_processing.py that crops all the images of the dataset to 24x24x3 images (sunglasses and no_sunglasses).
To do so, I have defined:
* *crop_dataset* that loads the json that contains info (presence of sunglasses, filename, bbox, etc.) of all the faces (in all images), creates a new folder where to save all the cropped images and then, I go through both dataset folder (img & no_sunglasses), ensure that I get non-empty images (not Nonetype) and crop them thanks to the bbox keys corresponding to finally, save them. I print a message every 50 cropped images to be aware of the lasting time.

For the cropped images, I need to know more about their shapes to resize them later and the distribution of labels (sunglasses vs. no sunglasses). 
To do so,  I have defined:
* *get_shapes* extracts all the cropped images and extract each sample's width & height and save them all in widths & heights lists 
* *get_labels_info* that counts the amount of no_sunglasses and sunglasses images in the dataset and then, returns a list of labels list and positive & negative images rate in the dataset.__
Also, I plot the distributions of heights, widths after *get_shapes* (min_width = 23 and min_height = 24) and I plot the ditribution of labels after *get_labels_info* and I get that 98.5% of no_sunglasses (very very imbalanced dataset)

For the tuning of Deep Learning models (convolutional_sunglasses_detection.py and dense_sunglasses_detection.py), the objective is to tune a selected architecture (fixed type/number of layers) with the following parameters: f (number of conv2D/maxpool2D), d (number of dense layers) & batch size (number of samples that will be propagated through the network). 
To do so, I have defined: 
* *open_resize* to extract an image from a path and adds it to a list, 
* *build_folder* that creates a folder with a chosen name & path, 
* *globalize* that make path in relation with an existing directory, 
* *build_dataset* that retruns images & labels dataset given an amount of no_sunglasses images (which are a majority in the initial dataset), 
* *separate_test_train* that returns training & testing sets from the dataset and TEST_PROPORTION, 
* *labelize* that changes a vector softmax output to a categorical vector (from [0.3, 0.7] to [0, 1]), 
* *plot_epoch* that takes history of model fitting and some metrics and that saves plots of each metric in a chosen filename, 
* *multiple_append* that append multiple elements to multiple lists and finally, 
* *build_model* that takes no argument and do:
  * create folder for callbacks & plots savings
  * create scores, col_f, col_d, col_batch, conv, dense & execution_time containing: f1-score, f value, d value, batch size, number of Conv2D, number of Dense & prediction time
  * train a chosen architecture for each (f,d,batch size), then:
    * saves csv logger (saves the tracked metrics during training), evolution of recall & precision during training, model file (in hdf5), 
    * compute time for prediction and f1-score for each model
    * saves an excel tab with columns containing network parameters and his performances (score, execution time)
