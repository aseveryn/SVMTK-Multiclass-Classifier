import sys
import os
import subprocess
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SVM_HOME = "SVM-Light-1.5-rer"
SVM_LEARN = os.path.join(SVM_HOME, "svm_learn")
SVM_TEST = os.path.join(SVM_HOME, "svm_classify")


class SVMTK:
  def __init__(self, params):
    self.params = params

  def train(self, train, model, quite=False):
    cmd = [SVM_LEARN] + self.params.split() + [train] + [model]
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
    cmd = [SVM_TEST, test, model, pred]
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
  mc = SVMTK(params=opts.tk_params)
  if opts.train:
    mc.train(train_file, model, quite=opts.quite)
  if opts.test:
    mc.test(test_file, model, pred, quite=opts.quite)
  if opts.eval:
    mc.eval(test_file, pred)
  else:
    mc.train(train_file, model, quite=opts.quite)
    mc.test(test_file, model, pred, quite=opts.quite)


if __name__ == '__main__':
  main()
  

