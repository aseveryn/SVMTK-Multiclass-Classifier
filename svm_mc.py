from __future__ import division
import os
import re
import random
import logging
import multiprocessing
from collections import defaultdict
from sklearn import metrics
from svm import SVMTK, LibLinear, LibSVM

SVMTK_FVEC_PAT = re.compile(r"(?P<label>\S+)\s+.*\|(ET|BV)\| (?P<vector>.*) \|EV\|")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

random.seed(123)
POS = "+1"
NEG = "-1"
NFOLDS = 5


def unwrap_self_svmtrain(arg, **kwarg):
  return MultiClassSVM.svm_train(*arg, **kwarg)


def unwrap_self_svmtest(arg, **kwarg):
  return MultiClassSVM.svm_test(*arg, **kwarg)


class MultiClassSVM:
  def __init__(self, 
               train, 
               binary_svm,
               ova_dir="ova", 
               params="-t 5 -C V", 
               ncpus=None, 
               quite=True):
    self.dir = os.path.dirname(train)
    self.ova_path = os.path.join(self.dir, ova_dir)
    self.ncpus = ncpus
    self.quite = quite
    self.svm = binary_svm
    self.params_str = "_".join(params.split())
    self.model_suff = self.params_str + ".model"
    self.pred_suff = self.params_str + ".pred"

    examples = []
    cat2ex = defaultdict(list)
    if not isinstance(self.svm, SVMTK):
      train = self.convert_svmtk_to_svmlight(train)

    for line in open(train):
      label, vec = line.strip().split(" ", 1)
      ex = (label, vec)
      examples.append(ex)
      cat2ex[label].append(ex)
      
    self.cat2ex = cat2ex
    self.categories = cat2ex.keys()
    self.examples = examples
    self.num_categories = len(self.categories)
    self.num_examples = len(self.examples)

  def svm_train(self, train, model, quite):
    self.svm.train(train, model, quite)

  def svm_test(self, test, model, pred, quite):
    self.svm.test(test, model, pred, quite=quite)

  def print_stats(self):
    print "Num examples", self.num_examples
    print "Num categories", self.num_categories
    for label, examples in self.cat2ex.iteritems():
      print "{}: {}".format(label, len(examples))

  def create_ova_data(self, mode="train"):
    if not os.path.exists(self.ova_path):
      os.makedirs(self.ova_path)
    for i, cat in enumerate(self.categories):
      out_fname = "{}.{}".format(cat, mode)
      out_path = os.path.join(self.ova_path, out_fname)
      logging.info("Writing ova file for label {} to: {}".format(cat, out_path))
      self.write_ova_examples(self.examples, cat, out_path)

  def get_true_labels(self, labels_file):
    y_true = []
    for line in open(labels_file):
      label = line.strip().split(" ", 1)[0]
      y_true.append(label)
    return y_true

  def write_ova_examples(self, examples, category, out_file):
    with open(out_file, "w") as out:
      for label, ex in examples:
        if label == category:
          label = POS
        else:
          label = NEG
        out.write("{} {}\n".format(label, ex))

  def iterate_svm_files(self, file_types=["train", "model"]):
    for cat in self.categories:
      print "Category", cat
      files = (os.path.join(self.ova_path, "{}.{}".format(cat, ftype)) for ftype in file_types)
      yield files

  def _extract_fvec_from_svmtk_file(self, fname, outfname):
    with open(outfname, "w") as out:
      for line in open(fname):
        match = SVMTK_FVEC_PAT.match(line.strip())
        if match:
          label = match.group("label")
          fvec = match.group("vector")
          out.write("{} {}\n".format(label, fvec))
        else:
          raise TypeError("Failed to parse SVM-TK example. Check formatting: {}".format(line))

  def convert_svmtk_to_svmlight(self, fname):
    ex = open(fname).readline().strip()
    match = SVMTK_FVEC_PAT.match(ex)
    if match:
      outfname = fname + ".fvec.liblinear"
      self._extract_fvec_from_svmtk_file(fname, outfname)
      fname = outfname
    return fname

  def _train(self):
    if self.ncpus:
      pool = multiprocessing.Pool(self.ncpus)
      args = [(self, train, model, self.quite) for train, model 
        in self.iterate_svm_files(["train", self.model_suff])]
      pool.map(unwrap_self_svmtrain, args)   
      pool.terminate()
    else:
      for (train, model) in self.iterate_svm_files(["train", self.model_suff]):
        self.svm_train(train, model, quite=self.quite)

  def train(self):
    self.create_ova_data()
    self._train()
  

  def test(self, test_file):
    if not isinstance(self.svm, SVMTK):
      test_file = self.convert_svmtk_to_svmlight(test_file)
      # Liblinear doesn't accept string labels (so need to replace them with 1.0)
      if isinstance(self.svm, LibLinear):
        nolabel_test_file = test_file + ".nolabel"
        with open(nolabel_test_file, "w") as out:
          for line in open(test_file):
            label, ex = line.strip().split(" ", 1)
            out.write("1.0 {}\n".format(ex))
        test_file = nolabel_test_file

    if self.ncpus:
      pool = multiprocessing.Pool(self.ncpus)
      args = [(self, test_file, model, pred, self.quite) for (model, pred) 
        in self.iterate_svm_files([self.model_suff, self.pred_suff])]
      pool.map(unwrap_self_svmtest, args)   
      pool.terminate()
    else:
      for (model, pred) in self.iterate_svm_files([self.model_suff, self.pred_suff]):
        self.svm_test(test_file, model, pred, quite=self.quite)

  def eval(self, labels_file=None):
    predictions = defaultdict(list)
    for cat in self.categories:
      # if cat == "neutral": continue
      pred_fname = "{}.{}".format(cat, self.pred_suff)
      pred = os.path.join(self.ova_path, pred_fname)
      for i, line in enumerate(open(pred)):
        score = float(line.strip())
        predictions[i].append((score, cat))

    y_pred = [max(predictions[i])[1] for i in xrange(len(predictions))]

    with open(os.path.join(self.ova_path, "all." + self.pred_suff), "w") as out:
      for label in y_pred:
        out.write("{}\n".format(label))

    if not labels_file:
      labels_file = os.path.join(self.ova_path, "labels.test")
    y_true = self.get_true_labels(labels_file)

    self.print_metrics(y_true, y_pred)
      

  def train_test_eval(self, test):
    self.train()
    self.test(test)
    self.eval(test)

  def print_metrics(self, y_true, y_pred, print_averages=True):
    print
    print "{:^30}".format("Confusion matrix")
    categories = sorted(self.categories)
    labels = " ".join("{:>10}".format(c) for c in categories)
    print "{:>10} {} {:>10}".format("gold\pred", labels, "total")
    for cat, predictions in zip(categories, metrics.confusion_matrix(y_true, y_pred)):
      vals = " ".join("{:>10d}".format(p) for p in predictions)
      print "{:>10} {} {:>10}".format(cat, vals, sum(predictions))
    print

    acc = metrics.accuracy_score(y_true, y_pred)
    print "Accuracy: {:.4f}".format(acc) 
    
    idx = 0
    d = {}
    for l in self.categories:
      d[l] = idx
      idx += 1

    print metrics.classification_report([d[y] for y in y_true], 
                                        [d[y] for y in y_pred], 
                                        target_names=self.categories)

    if print_averages:
      print "Macro averaging"
      self._print_metrics(y_true, y_pred, average='macro')

      print "Micro averaging"
      self._print_metrics(y_true, y_pred, average='micro')

  def _print_metrics(self, y_true, y_pred, average='macro'):
    precision = metrics.precision_score(y_true, y_pred, average=average)
    print "Precision: {:.4f}".format(precision) 
    
    recall = metrics.recall_score(y_true, y_pred, average=average)
    print "Recall: {:.4f}".format(recall) 
    
    f1 = metrics.f1_score(y_true, y_pred, average=average) 
    print "F1: {:.4f}".format(f1) 


def test_ova():
  train = "/Users/aseveryn/PhD/soft/SVMS/jlis-0.5/data/multiclass/small.train"
  test = "/Users/aseveryn/PhD/soft/SVMS/jlis-0.5/data/multiclass/small.test"
  mc = MultiClassSVM(train, ncpus=2)
  mc.train_test_eval(test)
  

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

  op.add_option("--ova_dir",
                default="ova",
                help="input directory with OVA svm files")

  op.add_option("--ncpus",
                type=int, default=None,
                help="number of CPUs.")

  op.add_option("--params", default="-t 5 -C V", 
                help="paramerters for SVM-TK")

  op.add_option("--svm_folder", default=None, 
                help="folder with svm.train and svm.test")

  op.add_option("--svm", default="svmtk", 
                help="back-end binary svm to use: [svmtk,liblinear]")

  op.add_option("--optimize_j",
                action="store_true", default=False,
                help="optimize -j parameter (only for SVMTK)")


  (opts, args) = op.parse_args()
  
  if opts.svm_folder:
    train_file = os.path.join(opts.svm_folder, "svm.train")
    test_file = os.path.join(opts.svm_folder, "svm.test")
  else:
    if len(args) != 2:
        op.print_help()
        op.error("this script takes exactly two argument.")
    train_file = args[0]
    test_file = args[1]


  if opts.svm == "svmtk":
    svm = SVMTK(opts.params, optimize_j=opts.optimize_j)
  elif opts.svm == "liblinear":
    svm = LibLinear(opts.params)
  elif opts.svm == "libsvm":
    svm = LibSVM(opts.params)
  else:
    raise TypeError("Unsupported binary svm type %s" % opts.svm)

  mc = MultiClassSVM(train_file, 
                     binary_svm=svm,
                     params=opts.params, 
                     ova_dir=opts.ova_dir, 
                     ncpus=opts.ncpus, 
                     quite=opts.quite)
  mc.print_stats()
  if opts.train:
    mc.train()
  if opts.test:
    mc.test(test_file)
  if opts.eval:
    mc.eval(test_file)
  if not (opts.train or opts.test or opts.eval):
    mc.train_test_eval(test_file)


if __name__ == '__main__':
  main()
  

