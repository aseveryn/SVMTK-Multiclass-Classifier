
all:
	cd SVM-Light-1.5-rer; make
	cd liblinear-1.93; make


clean:
	cd liblinear-1.93; make clean
	cd SVM-Light-1.5-rer; make clean

