#SVM for Image Classification

##How to use:

>python SVM.py training_data test_data linear

or 

>python SVM.py training_data test_data rbf

Where 'training_data' is the name of the folder in this directory where the training
examples are stored and 'test_data' is the folder of examples used for testing against. Note that the rbf does a grid search for parameters 
so it may take a minute or two to run, depending on the number of processing cores on the computer. 
