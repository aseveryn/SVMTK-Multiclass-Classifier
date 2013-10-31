SVMTK-Multiclass-Classifier
===========================

## Description
Python wrapper around SVM-TK binary classifier to perform multiclass classification

## Install
requires:
	Python 2.6+ (sklearn, numpy)
	gcc (to compile SVM-Light-TK)

The missing packages can be installed via conventional python package managers, e.g. easy_install or pip:

* pip install numpy
* pip install scikit-learn

Building:

* build SVM-Light-TK: go to folder SVM-Light-1.5-rer and type:
	make

## Usage

### Multi-class classification
To run train and test a multi-class classifier:

	python svm_mc.py [options] svm.train svm.test

where svm.train and svm.test are learning and test files in the SVM-Light/SVM-Light-TK format, where the target is a string label of the class, for example:

	label1  id1:value1 id2:value2 id3:value3…

To run a demo execute:

	python svm_mc.py --params="-t 0" --ncpus=2 svm.train svm.test

which builds a one-vs-all multiclass classifier using SVM-TK as a back-end binary classifier. The **--params** option specify a linear kernel (-t 0) and runs in parallel using 2 cpus (--ncpus=2) to train and test models for individual classes. Finally, it prints a confusion matrix and a per-class performance table (Precision, Recall and F-1). For more details on usage, type:
	python svm_mc.py -h

### Cross validation
To perform cross-validation on a single dataset use the following:

	python svm_mc_cv.py --nfolds=5 svm.data

which is going to automatically split the svm.data into 5 train/test folds, train a multi-class classifer on each fold. Finally it will report the accuracy on each fold and the averaged accuracy across all folds.

## Extensions:

**svm_mc.py** uses **svm.py** which wraps a C binary of SVM-Light-TK, LibSVM and LibLinear using **subprocess** module. 

It can be further extended to plug in other backend binary classifiers implemented in any other language, e.g. AdaBoost, Logistic Regression, etc.

	