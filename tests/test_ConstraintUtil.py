import sys
from pathlib import Path
import logging
from unittest import mock, TestCase, main, skip
from ConstraintUtil import ConstraintUtil
from LogitUtil import logit
from FileUtil import FileUtil
import time
from datetime import date
from collections import namedtuple
from YamlUtil import YamlUtil
from collections import Counter

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
"""

class KillerReader(YamlUtil):
    def __init__(self, *args, **kwargs):
        super(KillerReader, self).__init__(*args, **kwargs)

    def empty_pigeonhole(self, count_dict: dict):
        s = Sudoku(rows=9)
        for row in range(1,10):
            for col in range(1,10):
                var_name = s.var_name(row, col)
                if count_dict[var_name] < 2:
                    logger.warning(f'Missing: {var_name}')
                elif count_dict[var_name] > 2:
                    logger.warning(f'duplicated: {var_name}')

    def pigeonhole(self):
        s = Sudoku(rows=9)
        logger.debug (f'as namedtuple: {self.asnamedtuple}')
        s_orig = s.variables()
        covered = []
        for v in self.asdict.values():
            covered.extend(v)
        logger.debug (f'There were {len(covered)} cells covered: {covered}.')

        covered.extend(s_orig) # Should have 2 of every variable in covered.

        covered_count = Counter(covered)
        logger.debug(f'Covered dictionary: {covered_count}')
        self.empty_pigeonhole(covered_count)


class Sudoku:
    def __init__(self, rows:int=3, cols:int=None, offset:int=1):
        self._rows = rows # normally 9, but in suko it's 3
        self._cols = cols or self._rows # let's assume square.
        self._offset = offset # normally 1

    def var_name(self, row:int, col:int, prefix:str=None):
        if prefix:
            return f'{prefix}{row:1d}{col:1d}'
        else:
            return row*10 + col

    def variables(self):
        ans = []
        for row in range(self._offset, self._rows + self._offset):
            for col in range(self._offset, self._cols + self._offset):
                ans.append(self.var_name(row, col))
        return ans

    def all_rows(self):
        rows = []
        for r in range(self._offset, self._rows + self._offset):
            row = []
            for col in range(self._offset, self._cols + self._offset):
                row.append(self.var_name(r, col))
            rows.append(row)
        return rows

    def all_cols(self):
        cols = []
        for r in range(self._offset, self._rows + self._offset):
            col = []
            for c in range(self._offset, self._cols + self._offset):
                col.append(self.var_name(c, r))
            cols.append(col)
        return cols

    def all_3x3_boxes(self):
        boxes = [
        [11, 12, 13, 21, 22, 23, 31, 32, 33],
        [41, 42, 43, 51, 52, 53, 61, 62, 63],
        [71, 72, 73, 81, 82, 83, 91, 92, 93],
        [14, 15, 16, 24, 25, 26, 34, 35, 36],
        [44, 45, 46, 54, 55, 56, 64, 65, 66],
        [74, 75, 76, 84, 85, 86, 94, 95, 96],
        [17, 18, 19, 27, 28, 29, 37, 38, 39],
        [47, 48, 49, 57, 58, 59, 67, 68, 69],
        [77, 78, 79, 87, 88, 89, 97, 98, 99],
        ]
        return boxes

    def print_solution(self, solution:dict):
        for row in range(self._offset, self._rows + self._offset):
            for col in range(self._offset, self._cols + self._offset):
                print (f' {solution[self.var_name(row, col)]} ',end=" ")
            print(" ")

    def print_solutions(self, solution_list:list):
        for row in range(self._offset, self._rows + self._offset):
            for col in range(self._offset, self._cols + self._offset):
                answers = set()
                for solution in solution_list:
                    answers.add(solution[self.var_name(row, col)])
                if (len(answers) == 1):
                    print (f' {solution[self.var_name(row, col)]} ',end=" ")
                else:
                    print (f' 0 ',end=" ")
            print(" ")



class Test_ConstraintUtil(TestCase):
    def setUp(self):
        pass

    @logit()
    def test_sudoku(self):
        sbox = Sudoku(rows=9)
        v = sbox.variables()
        cu = ConstraintUtil()
        # Define the variables: 9 rows of 9 variables range in 1..9
        cu.variables(v, range(1,10))

        rows = sbox.all_rows()
        for row in rows:
            cu.constraint_all_different(row)

        cols = sbox.all_cols()
        for col in cols:
            cu.constraint_all_different(col)

        # Each 3x3 box has different values
        boxes = sbox.all_3x3_boxes()
        for box in boxes:
            cu.constraint_all_different(box)

        # Some values are given.
        initValue = [
            [0, 9, 0, 7, 0, 0, 8, 6, 0],
            [0, 3, 1, 0, 0, 5, 0, 2, 0],
            [8, 0, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 7, 0, 5, 0, 0, 0, 6],
            [0, 0, 0, 3, 0, 7, 0, 0, 0],
            [5, 0, 0, 0, 1, 0, 7, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 9],
            [0, 2, 0, 6, 0, 0, 0, 5, 0],
            [0, 5, 4, 0, 0, 8, 0, 7, 0],
        ]
        cu.set_initial(initValue)

        solutions = cu.solve()
        print (f'There were {len(solutions)} solutions.')
        sbox.print_solutions(solutions)

    def test_suko(self):
        sbox = Sudoku(rows=3)
        v = sbox.variables()
        cu = ConstraintUtil()
        cu.variables(v, range(1,10))
        cu.constraint_all_different(v)
        cu.constraint_exact_sum([11,12,21,22], 28) # top left
        cu.constraint_exact_sum([31,32,21,22], 18) # bottom left
        cu.constraint_exact_sum([12,13,22,23], 20) # top right
        cu.constraint_exact_sum([32,33,22,23], 21) # bottom right

        cu.constraint_exact_sum([11,12,13], 18)
        cu.constraint_exact_sum([32,33], 11)

        solutions = cu.solve()
        print (f'There were {len(solutions)} solutions.')
        for i, solution in enumerate(solutions):
            print (f'Solution {i}')
            sbox.print_solution(solution)

    # @skip("too long! skipping")
    def test_killer_sudoku(self):
        sbox = Sudoku(rows=9)
        v = sbox.variables()
        cu = ConstraintUtil()
        cu.variables(v, range(1,10))

        rows = sbox.all_rows()
        for row in rows:
            cu.constraint_all_different(row)

        cols = sbox.all_cols()
        for col in cols:
            cu.constraint_all_different(col)

        # Each 3x3 box has different values
        boxes = sbox.all_3x3_boxes()
        for box in boxes:
            cu.constraint_all_different(box)

        k = KillerReader('./killer22Aug20.yml')
        k.pigeonhole()
        puzzle_sum = 0
        for key, value in k.asdict.items():
            # key, value is like a23, [11,21,31]
            sum = int(key[1:])
            logger.debug(f'cells {value} have a sum of {sum:02d}')
            cu.constraint_exact_sum(value, sum)
            puzzle_sum += sum
        sudoku_sum = 45 * 9
        logger.info(f'puzzle sum is {puzzle_sum}. Theoretical sum is {sudoku_sum}')

        # Here are a few more constraints (by inspection)
        # cu.constraint_in_set([12,13], [8,9]) # con 1. because that's the only way to get 17
        # cu.constraint_in_set([28,29], [1,2,3,5,6,7]) # con 2. many ways to get 8, but not 4+4
        # cu.constraint_in_set([51,52], [3,4,5,7,8,9]) # con 3. many ways to get 12, but not 6+6. Took 42 min for con 1-3
        # cu.constraint_in_set([17,18,19], [7,6,5,4]) # con 4. can't be 8 or 9 (prev constraint) and must add to 17. took 25 min for con 1-4.
        # cu.constraint_in_set([48,49], [1,2]) # con 5. Only way to get 3. Took 24 min for con 1-5
        # cu.constraint_in_set([58,59], [4,3]) # con 6. Sum 7 but 1 and 2 removed from con 5. Took 24 min.
        # cu.constraint_in_set([88,98], [9,8,7,6]) # con 7. Only way to get 15. Took 18 min.
        # cu.constraint_in_set([61,62,71,72], [1,2,3,4]) # only way to make 10
        # cu.constraint_in_set([87,97,96], [1,2,3,4,5])
        # cu.constraint_in_set([64, 65], [1,2,3,4,5])
        # cu.constraint_in_set([63,73,83], [9,8,7]) # only way to make 24
        # cu.constraint_in_set([13,2333,43,53,93], [1,2,3,4,5,6]) # rest of column can't be 9,8, or 7.
        # Helpers for 2Jul20
        # cu.constraint_in_set([11,12], [1,2,3,4,5,6]) # Several ways to get 8
        # cu.constraint_in_set([15,16], [1,2,3,4,5,7,8,9]) # Several ways to get 12, but not 6+6


        start_time = time.time()
        logger.debug (f'Starting to solve.')
        solutions = cu.solve()
        time_in_seconds = time.time() - start_time
        logger.debug (f'Found {len(solutions)} solutions in {(time.time() - start_time)/60.0:.1f} minutes.')

        if len(solutions) == 1:
            logger.info(f'Sole solution is:\n')
            sbox.print_solution(solutions[0])
        else:
            sbox.print_solutions(solutions)
