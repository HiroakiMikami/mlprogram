from importlib import import_module
import sys

module_name = sys.argv[1]
sys.argv.pop(1)
module = import_module(module_name)
