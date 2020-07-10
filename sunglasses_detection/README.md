For the tuning of the Deep Learning models, the objective is to tune a selected architecture with the following parameters: f (number of conv2D/maxpool2D), d (number of dense layers) & batch size (number of samples that will be propagated through the network). 

To do so, I have defined: 
open_resize to extract an image from a path and adds it to a list, 
build_folder that creates a folder with a chosen name & path, 
globalize that make path in relation with an existing directory, 
build_dataset that retruns images & labels dataset given an amount of no_sunglasses images (which are a majority in the initial dataset), 
separate_test_train that returns training & testing sets from the dataset and TEST_PROPORTION, 
labelize that changes a vector softmax output to a categorical vector (from [0.3, 0.7] to [0, 1]), 
plot_epoch that takes history of model fitting and some metrics and that saves plots of each metric in a chosen filename, 
multiple_append that append multiple elements to multiple lists and finally, 
build_model that takes no argument and train a chosen architecture for different (f,d,batch size) and 
saves csv logger, metrics curves evolution wrt epochs, model (in hdf5), time for prediction and f1-score to end by saving an excel tab with columns containing network parameters and his performances (score, execution time).
