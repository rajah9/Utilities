import sys
from inspect import currentframe, getfile
from os.path import realpath, abspath, split

curr_folder = realpath(abspath(split(getfile(currentframe()))[0])) # Should be this Utilities dir
if curr_folder not in sys.path:
    sys.path.insert(0, curr_folder)
