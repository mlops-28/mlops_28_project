import os

_ARTSY_ROOT = os.path.abspath(os.path.dirname(__file__))  # root of artsy folder
_PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(_ARTSY_ROOT)))  # root of project

_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_CONFIGS = os.path.join(_PROJECT_ROOT, "configs")
