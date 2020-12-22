import unittest
import sys
import pathlib


sys.path.append(str(pathlib.Path.cwd().parent))

import climate_ml.preprocessing as preproc


class TestPreproc(unittest.TestCase):
    def teast_covariance(self):
        