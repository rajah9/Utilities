from pathlib import Path, PurePath
from sys import path
from typing import Union

class Add_path():
    @classmethod
    def add_path(cls, newPath: Union[str, PurePath]) -> list:
        """
        Add the given path to the sys.path.
        Typical call:
          parent = Path('..').resolve() # to add the parent dir
          Add_path.add_path(parent)
        :param newPath:  str or pathlib.PurePath of new path
        :return: full sys.path array
        """
        strPath = str(newPath) if isinstance(newPath, PurePath) else newPath

        if strPath in path:
            print(f'path: {newPath} is already on sys.path. (No action taken.)')
            return path
        else:
            print(f'Adding new path: {strPath} to sys.path.')
            path.append(strPath)
        return path

    @classmethod
    def add_parent(cls):
        """
        Add the parent dir to sys.path.
        Typical call:
          Add_path.add_parent()
        :return: full sys.path array
        """
        parent = Path('..').resolve()  # Must add the parent dir of Utilities.
        print (f'parent dir is {parent}')
        return cls.add_path(parent)