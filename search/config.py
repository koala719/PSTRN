# -*- coding: UTF-8 -*-

# Author: Perry
# @Time: 2019/10/23 15:05


import os

_project_root_path = os.path.dirname(os.path.realpath(__file__))


class Config(object):

    def __init__(self):
        pass

    # The Root of Whole Project
    @property
    def root_path(self):
        return _project_root_path

    # The Root of Metrics that need to be saved, like model, training records
    @property
    def save_root(self):
        return os.path.join(proj_cfg.root_path, "save_multi_P")


proj_cfg = Config()
