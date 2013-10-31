import sys
from sklearn import metrics



class Eval:
  def __init__(self, pred, test):
    
    y_true = []
    for line in open(test):
      label, _ = line.strip().split(" ", 1)
      y_true.append(label)
    self.y_true = y_true
    self.categories = list(set(y_true))
    self.y_pred = [line.strip() for line in open(pred)]

  def print_confusion_matrix(self, y_true, y_pred):
    print
    print "{:^30}".format("Confusion matrix")
    categories = sorted(self.categories)
    labels = " ".join("{:>10}".format(c) for c in categories)
    print "{:>10} {} {:>10}".format("gold\pred", labels, "total")
    for cat, predictions in zip(categories, metrics.confusion_matrix(y_true, y_pred)):
      vals = " ".join("{:>10d}".format(p) for p in predictions)
      print "{:>10} {} {:>10}".format(cat, vals, sum(predictions))
    print

  def print_metrics(self, 
                    print_confusion_matrix=False, 
                    print_averages=False):
    y_true = self.y_true
    y_pred = self.y_pred

    if print_confusion_matrix:
      self.print_confusion_matrix(y_true, y_pred)

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

if __name__ == '__main__':
  pred = sys.argv[1]
  test = sys.argv[2]

  ev = Eval(pred, test)
  ev.print_metrics()