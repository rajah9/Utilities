from constraint import Problem, AllDifferentConstraint, ExactSumConstraint, InSetConstraint
import sys

from Util import Util

class ConstraintUtil(Util):
    def __init__(self):
        super(ConstraintUtil, self).__init__(null_logger=False)
        self.problem = Problem()
        self._hasVariables = False
        self._hasConstraints = False

    def variables(self, var_list:list, possible: list):
        if isinstance(possible, range):
            self.problem.addVariables(var_list, possible)
        elif isinstance(possible, list):
            for v, p in zip(var_list, possible):
                self.problem.addVariable(v, p)
        else:
            self.logger.error(f'Domain must be a range or a list of lists, but instead is a {type(possible)}')
            return
        self._hasVariables = True
        return

    def constraint_all_different(self, list_of_vars:list):
        self.problem.addConstraint(AllDifferentConstraint(), list_of_vars)
        self._hasConstraints = False

    def constraint_exact_sum(self, list_of_vars:list, sum:int):
        self.problem.addConstraint(ExactSumConstraint(sum), list_of_vars)

    def constraint_in_set(self, list_of_vars: list, set_as_list: list):
        self.problem.addConstraint(InSetConstraint(set_as_list), list_of_vars)

    def solve(self):
        problem = self.problem

        # Get the solutions.
        solutions = problem.getSolutions()
        return solutions

    def set_initial(self, initValue:list):
        for i in range(1, 10):
            for j in range(1, 10):
                if initValue[i - 1][j - 1] != 0:
                    self.constraint_in_set(list_of_vars=[ (i * 10 + j)], set_as_list=[initValue[i - 1][j - 1]])
                    # self.problem.addConstraint(
                    #     lambda var, val=initValue[i - 1][j - 1]: var == val, (i * 10 + j,)
                    # )


if __name__ == "__main__":
    cu = ConstraintUtil()
    solutions = cu.solve()
    # Print the solutions
    for solution in solutions:
        for i in range(1, 10):
            for j in range(1, 10):
                index = i * 10 + j
                sys.stdout.write("%s " % solution[index])
            print("")
        print("")


