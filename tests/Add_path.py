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
        :return: newly added path (or None)
        """
        strPath = str(newPath) if isinstance(newPath, PurePath) else newPath

        if strPath in path:
            print(f'path: {newPath} is already on sys.path. (No action taken.)')
        else:
            print(f'Adding new path: {strPath} to sys.path.')
            path.append(strPath)
        return strPath

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

    @classmethod
    def add_sibling(cls, subdir: str):
        """
        Add the sibling dir (child of parent dir) to sys.path.

        :param subdir:
        :return:
        """
        parent = Path('..')
        sib = parent / subdir
        if not sib.exists():
            print (f'sibling dir {sib.resolve()} does not exist. (Not adding)')
            return
        return cls.add_path(sib.resolve())

    @classmethod
    def get_ancestor(cls, how_far_back: int = 1, base_path: Union[str, PurePath] = None) -> PurePath:
        """
        Get the ancestor of the current directory.
        :param how_far_back: 0=this dir, 1=parent dir...
        :param base_path: The base path to use. if None, use current path.
        :return:
        """
        p = Path(base_path) if base_path else Path().resolve()
        print (f'base_path is {p}')
        if how_far_back <= 0:
            return p
        q = p.parents
        if how_far_back > len(q):
            print (f'Error: requesting ancestor that is {how_far_back} generations back but only have {len(q)}')
            return None
        return q[how_far_back - 1]

    @classmethod
    def get_child(cls, child_dir: str, base_path: Union[str, PurePath] = None) -> PurePath:
        p = Path(base_path) if base_path else Path().resolve()
        q = p.joinpath(child_dir)
        if not q.exists():
            print(f'Cannot find path {q.resolve()}. Returning None.')
            return None
        return q

    @classmethod
    def add_child(cls, child_dir: str, base_path: Union[str, PurePath] = None) -> PurePath:
        q = Add_path.get_child(child_dir=child_dir, base_path=base_path)
        if q:
            Add_path.add_path(q)
        else:
            print('child_dir does not exist')

    @classmethod
    def find_ancestor_with_child(cls, child: Union[str, PurePath], base_path: Union[str, PurePath] = None, search_from_leaf: bool = True):
        """
        Find the ancestor with the given child. Search from the current dir toward the root if search_from_leaf is True.
        :param child: PurePath or string of child to look for
        :param base_path: if None, use cwd. Otherwise, search from this path.
        :param search_from_leaf:
        :return:
        """
        p = Path(base_path) if base_path else Path().resolve()
        parents = p.parents
        parents_in_order = parents if search_from_leaf else parents.__reversed__()
        for parent in parents_in_order:
            test_dir = parent / child
            if test_dir.is_dir():
                return parent
        print(f'unable to find {child} in path of {p.resolve()}. Returning None')
        return None
