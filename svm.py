import sys
import os
import subprocess
import logging
from liblinear_predict import LibLinearModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

HOME="/Users/aseveryn/PhD/projects/svm-tk-mc"

class SVM:
  def __init__(self, params, binary_train, binary_test):
    self.params = params
    self.svm_learn = binary_train
    self.svm_test = binary_test

  def get_cmd_train(self, train, model, params):
    return [self.svm_learn] + self.params.split() + [train] + [model]

  def get_cmd_test(self, test, model, pred):
    return [self.svm_test, test, model, pred]

  def train(self, train, model, quite=False):
    cmd = self.get_cmd_train(train, model, self.params)
    logging.info("Training: {}".format(" ".join(cmd)))

    # Train
    if quite:
      proc_train = subprocess.Popen(cmd, stdout=subprocess.PIPE, close_fds=True)
      (stdout, stderr) = proc_train.communicate()
      train_log = model + ".train.log"
      with open(train_log, 'w') as log:
        log.write(stdout)
    else:
      proc_train = subprocess.Popen(cmd, stdout=sys.stdout, close_fds=True)
      proc_train.communicate()


  def test(self, test, model, pred, quite=False):
    # Classify
    cmd = self.get_cmd_test(test, model, pred)
    logging.info("Testing: {}".format(" ".join(cmd)))
    
    if quite:
      proc_test = subprocess.Popen(cmd, stdout=subprocess.PIPE, close_fds=True) 
      (stdout, stderr) = proc_test.communicate()  
      test_log = model + ".test.log"
      with open(test_log, 'w') as log:
        log.write(stdout)
    else:
      proc_test = subprocess.Popen(cmd, stdout=sys.stdout, close_fds=True) 
      proc_test.communicate()

  def eval(self, test, pred):
    # Classify
    from sklearn import metrics
    y_true = [int(score.strip().split(" ", 1)[0]) for score in open(test)]
    pred = [float(score.strip()) for score in open(pred)]
    y_pred = []
    for y in pred:
      if y >= 0:
        y = 1
      else:
        y = -1
      y_pred.append(y)
      
    # print metrics.classification_report(y_true, y_pred)
    acc = float(metrics.accuracy_score(y_true, y_pred))
    f1 = float(metrics.f1_score(y_true, y_pred))
    p = float(metrics.precision_score(y_true, y_pred))
    r = float(metrics.recall_score(y_true, y_pred))

    print "Accuracy: {0:.4f}".format(acc)
    print "F1: {0:.4f}".format(f1)
    print "Precision/Recal: {0:.4f}/{0:.4f}".format(p, r)


class SVMTK(SVM):
  SVM_HOME = os.path.join(HOME, "SVM-Light-1.5-rer")
  SVM_LEARN = os.path.join(SVM_HOME, "svm_learn")
  SVM_TEST = os.path.join(SVM_HOME, "svm_classify")

  def __init__(self, params, optimize_j=False):
    self.optimize_j = optimize_j
    SVM.__init__(self, params, self.SVM_LEARN, self.SVM_TEST)

  def get_optimal_j(self, train):
    num_pos = 0.0
    num_neg = 0.0
    for line in open(train):
      label, _ = line.strip().split(" ", 1)
      label = float(label)
      if label > 0:
        num_pos += 1.0
      else:
        num_neg += 1.0
    j = num_neg/num_pos
    return j

  # def train(self, train, model, quite=False):
  #   j = self.get_optimal_j(train)
  #   self.params = "{} -j {:.4f}".format(self.params, j)
  #   SVM.train(self, train, model, quite)

  def get_cmd_train(self, train, model, params):
    if self.optimize_j:
      j = self.get_optimal_j(train)
      return [self.svm_learn] + self.params.split() + ["-j", "{:.4f}".format(j)] + [train] + [model]
    else:
      return SVM.get_cmd_train(self, train, model, params)


class LibSVM(SVM):
  SVM_HOME = os.path.join(HOME, "libsvm-2.91")
  SVM_LEARN = os.path.join(SVM_HOME, "svm-train")
  SVM_TEST = os.path.join(SVM_HOME, "svm-predict")

  def __init__(self, params):
    SVM.__init__(self, params, self.SVM_LEARN, self.SVM_TEST)

  def get_cmd_test(self, test, model, pred):
    return [self.svm_test, '-b 1', test, model, pred]



class LibLinear(SVM):
  SVM_HOME = os.path.join(HOME, "liblinear-1.93")
  SVM_LEARN = os.path.join(SVM_HOME, "train")
  SVM_TEST = os.path.join(SVM_HOME, "predict")

  def __init__(self, params):
    SVM.__init__(self, params, self.SVM_LEARN, self.SVM_TEST)


  # def test(self, test, model, pred, quite=False):
  #   model = LibLinearModel(model)
  #   model.predict(test, pred)
  #   logging.info("Testing: {}".format(test))
    
  # def get_cmd_test(self, test, model, pred):
    # return [self.svm_test, test, model, pred]

def main():
  from optparse import OptionParser
  op = OptionParser()

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

  op.add_option("--tk_params", default="-t 5 -C V", 
                help="paramerters for SVM-TK")


  (opts, args) = op.parse_args()
  
  if len(args) != 2:
      op.print_help()
      op.error("this script takes only one argument.")

  train_file = args[0]
  test_file = args[1]
  base = os.path.splitext(train_file)[0]
  model = base + ".model" + "_".join(opts.tk_params.split())
  pred = base + ".pred"
  mc = LibLinear(params=opts.tk_params)
  if opts.train:
    mc.train(train_file, model, quite=opts.quite)
  if opts.test:
    mc.test(test_file, model, pred, quite=opts.quite)
  if opts.eval:
    mc.eval(test_file, pred)
  else:
    mc.train(train_file, model, quite=opts.quite)
    mc.test(test_file, model, pred, quite=opts.quite)



def test_extract_fvec():
  from svm_mc import SVMTK_FVEC_PAT
  ex = "negative |BT| (ROOT (S (ADVP (R (just)) (R (so)) (R (much)) (A (more))) (positive-NP (positive-N (fun))) (VP (P (to)) (V (watch))))) |ET| 67:2.0 76:0.3333333333333333 79:0.3333333333333333 85:2.0 131:0.3333333333333333 285:0.3333333333333333 328:0.3333333333333333 628:0.3333333333333333 696:0.3333333333333333 823:0.3333333333333333 23399:0.3333333333333333 |EV|"
  match = SVMTK_FVEC_PAT.match(ex)
  if match:
    print match.group("label")
    print match.group("vector")

  svm = SVMTK("-t 5")
  lib = LibLinear("-s 2")
  print isinstance(svm, SVMTK)


if __name__ == '__main__':
  # main()
  test_extract_fvec()
  

