import os
import logging
import shutil
import math
from collections import defaultdict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def make_dirs(path):
  if os.path.exists(path):
    logging.info("Removing existing directory: %s", path)
    shutil.rmtree(path)
  os.makedirs(path)

def normalize_fvec(fvec):
  # compute norm
  norm = 0.0
  for fval in fvec.itervalues():
    norm += fval*fval
  norm = math.sqrt(norm)
  if math.fabs(norm) < 1e-8:
    return
  # normalize each value
  for fid in fvec:
    fvec[fid] /= norm


def fvec2str(fvec):
  return " ".join(["{}:{}".format(fid, fval) for fid, fval in sorted(fvec.items())])

def build_fvec(words, alphabet, normalize=True):
  fvec = defaultdict(float)
  for word in words:
    idx = alphabet.add(word)
    fvec[idx] += 1.0
  if normalize:
    normalize_fvec(fvec)
  return fvec


class Alphabet:
  def __init__(self, start_feature_id=1):
    self.vocabulary = {}
    self.fid = start_feature_id

  def add(self, item):
    idx = self.vocabulary.get(item, None)
    if not idx:
      idx = self.fid
      self.vocabulary[item] = idx
      self.fid += 1
    return idx

  def __len__(self):
    return len(self.vocabulary)