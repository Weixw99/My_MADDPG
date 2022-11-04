import os.path as osp
import imp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
