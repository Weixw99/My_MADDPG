import os.path as osp
import imp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)  # 找到该文件名的绝对地址
    return imp.load_source('', pathname)
