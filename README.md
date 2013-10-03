SVMTK-Multiclass-Classifier
===========================

## Description
Python wrapper around SVM-TK binary classifier to perform multiclass classification

## Install
requires:
	Python 2.6+ (sklearn, numpy)
	gcc (to compile SVM-Light-TK)

usage:

	python svmtk_mc.py [options] svm.train svm.test

where svm.train and svm.test are learning and test files in the SVM-Light/SVM-Light-TK format, where the target is a string label of the class, for example:

	label1  id1:value1 id2:value2 id3:value3…

To run a demo:

* build SVM-Light-TK: go to folder SVM-Light-1.5-rer and type
	make
* to run traininig, testing and evaluation on the demo data

execute:

	python svmtk_mc.py --tk_params="-t 0" --ncpus=2 svm.train svm.test

which runs SVM-TK binary classifier with a linear kernel (-t 0) in parallel using 2 cpus (--ncpus=2) to train and test models for individual classes. Finally, it prints a confusion matrix and a per-class performance table (Precision, Recall and F-1). For more details on usage, type:
	python svmtk_mc.py -help


## Extensions:

**svmtk_mc.py** uses **svmtk.py** which wraps a C binary of SVM-Light-TK using **subprocess** module. 

It can be used as an example to plug in other classifiers implemented in any other language, e.g. AdaBoost, Logistic Regression, etc.

	