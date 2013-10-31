

def make_cv_folds(examples, nfolds=5):
  fold_size = len(examples)/nfolds
  # print fold_size
  for fold in xrange(nfolds-1):
    a = fold * fold_size
    b = (fold + 1) * fold_size
    train = examples[:a] + examples[b:]
    test = examples[a:b]
    yield (train, test)

  b = (fold + 1) * fold_size
  train = examples[:b]
  test = examples[b:]
  yield (train, test)