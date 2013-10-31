from __future__ import division
import os
import multiprocessing
import random
from collections import defaultdict
from sklearn import metrics

from svm_mc import MultiClassSVM, unwrap_self_svmtrain, unwrap_self_svmtest
from svm import SVMTK, LibLinear
from util import make_dirs
from cv import make_cv_folds


random.seed(854)


class MultiClassSVMCrossValidation(MultiClassSVM):
  def __init__(self, 
               train, 
               binary_svm,
               ova_dir="ova", 
               params="-t 5 -C V", 
               nfolds=5,
               ncpus=None, 
               quite=True):
    MultiClassSVM.__init__(self, 
                           train, 
                           binary_svm=binary_svm,
                           ova_dir=ova_dir, 
                           params=params, 
                           ncpus=ncpus, 
                           quite=quite)
    self.nfolds = nfolds

  def create_cv_folds(self, dir="ova.cv", nfolds=5):
    make_dirs(self.ova_path)
    # make cv folds for examples from each of the categories
    folds = [make_cv_folds(examples, nfolds) for _, examples in self.cat2ex.iteritems()]

    # strip (train, test) pairs from each of the folds
    for i, fold_data in enumerate(zip(*folds), 1):
      print "Fold", i
      fold_path = os.path.join(self.ova_path, "fold-{}".format(i))
      make_dirs(fold_path)

      train_examples = []
      test_examples = []
      for (train_fold, test_fold) in fold_data:
        train_examples.extend(train_fold)
        test_examples.extend(test_fold)

      labels_fname = os.path.join(fold_path, "labels.test")
      with open(labels_fname, "w") as out:
        for label, _ in test_examples:
          out.write("{}\n".format(label))

      for cat in self.categories:
        # training  
        train_fname = os.path.join(fold_path, "{}.train".format(cat))
        self.write_ova_examples(train_examples, cat, train_fname)
        
        # testing
        test_fname = os.path.join(fold_path, "{}.test".format(cat))
        self.write_ova_examples(test_examples, cat, test_fname)

  def train_test_eval(self):
    self.train(self.nfolds)
    self.test()
    self.eval()

  def iterate_folds(self):
    for fold in os.listdir(self.ova_path):
      fold_path = os.path.join(self.ova_path, fold)
      yield fold_path

  def iterate_svm_files(self, file_types=["train", "model"]):
    for fold_path in self.iterate_folds():
      for cat in self.categories:
        print "Category", cat
        files = (os.path.join(fold_path, "{}.{}".format(cat, ftype)) for ftype in file_types)
        yield files

  def train(self, nfolds=5, quite=True):
    self.create_cv_folds(self.ova_path, nfolds)
    self._train()
    # if self.ncpus:
    #   pool = multiprocessing.Pool(self.ncpus)
    #   args = [(self, train, model, quite) for train, model in self.iterate_svm_files(["train", "model"])]
    #   pool.map(unwrap_self_svmtrain, args)   
    #   pool.terminate()
    # else:
    #   for (train, model) in self.iterate_svm_files(["train", "model"]):
    #     self.svm.train(train, model, quite=quite)        

  def test(self, quite=True):
    if self.ncpus:
      pool = multiprocessing.Pool(self.ncpus)
      # args = [(self, test, model, pred, self.quite) for (test, model, pred) in self.iterate_svm_files(["test", "model", "pred"])]
      args = [(self, test, model, pred, self.quite) for (test, model, pred) 
            in self.iterate_svm_files(["test", self.model_suff, self.pred_suff])]
      pool.map(unwrap_self_svmtest, args)   
      pool.terminate()
    else:
  #     for (test, model, pred) in self.iterate_svm_files(["test", "model", "pred"]):
      for (test, model, pred) in self.iterate_svm_files(["test", self.model_suff, self.pred_suff]):
        self.svm.test(test, model, pred, quite=quite)

  def eval(self):
    accuracies = []
    for i, fold_path in enumerate(self.iterate_folds()):
      fold_str = "FOLD: {}".format(i+1)
      print "="*len(fold_str)
      print fold_str  
      print "="*len(fold_str)
      predictions = defaultdict(list)
      for cat in self.categories:
        pred_fname = "{}.{}".format(cat, self.pred_suff)
        pred = os.path.join(fold_path, pred_fname)
        for i, line in enumerate(open(pred)):
          score = float(line.strip())
          predictions[i].append((score, cat))

      y_pred = [max(predictions[i])[1] for i in xrange(len(predictions))]

      labels_file = os.path.join(fold_path, "labels.test")
      y_true = [line.strip() for line in open(labels_file)]

      self.print_metrics(y_true, y_pred)
      acc = metrics.accuracy_score(y_true, y_pred)
      accuracies.append(acc)

    avg_acc = "Accuracy (avg across {} folds): {:.4f}".format(self.nfolds, sum(accuracies)/len(accuracies)) 
    print "="*len(avg_acc)
    print avg_acc



def main():
  from optparse import OptionParser

  usage = "usage: %prog [options] <svm.train> <svm.test>"
  op = OptionParser(usage=usage)

  op.add_option("--train",
                action="store_true", default=False,
                help="Train only")

  op.add_option("--test", 
                action="store_true",default=False,
                help="Test only")

  op.add_option("--eval",
                action="store_true", default=False,
                help="Evaluate only")

  op.add_option("-q", "--quite",
                action="store_true", default=False,
                help="quite output")
  
  op.add_option("--nfolds",
                action="store", 
                type=int, 
                default=5,
                help="Number of folds for cross-validation")

  op.add_option("--ova_dir",
                default="ova",
                help="input directory with OVA svm files")

  op.add_option("--ncpus",
                type=int, default=None,
                help="number of CPUs.")

  op.add_option("--params", default="-t 5 -C V", 
                help="paramerters for SVM-TK")

  op.add_option("--optimize_j",
                action="store_true", default=False,
                help="optimize -j parameter (only for SVMTK)")

  op.add_option("--svm", default="svmtk", 
                help="back-end binary svm to use: [svmtk,liblinear]")

  (opts, args) = op.parse_args()
  
  if len(args) != 1:
      op.print_help()
      op.error("this script takes exactly two argument.")
  svm_file = args[0]

  if opts.svm == "svmtk":
    svm = SVMTK(opts.params, optimize_j=opts.optimize_j)
  elif opts.svm == "liblinear":
    svm = LibLinear(opts.params)
  else:
    raise TypeError("Unsupported binary svm type %s" % opts.svm)

  mc = MultiClassSVMCrossValidation(svm_file, 
                     binary_svm=svm,
                     params=opts.params, 
                     nfolds=opts.nfolds,
                     ova_dir=opts.ova_dir, 
                     ncpus=opts.ncpus, 
                     quite=opts.quite)
  mc.print_stats()
  if opts.train:
    mc.train()
  if opts.test:
    mc.test()
  if opts.eval:
    mc.eval()
  if not (opts.train or opts.test or opts.eval):
    mc.train_test_eval()


if __name__ == '__main__':
  main()
  

