# csp
# set of variables, domain of each variable, constraints
# can have only inequality constraints (sufficient for map coloring problems)
class CSP:
    def __init__(self, domains, constraints):
        # Initialize the CSP with domains and constraints

        # key (variables) = regions
        # value (domain) = possible colors
        self.domains = domains

        # key = variables
        # value = values it should not be equal to
        self.constraints = constraints

        # key = variables
        # value = current assigned value
        self.assignments = {}
        for k in domains.keys():
            self.assignments[k] = None

        # all variables without an assigned value
        self.unassigned = domains.keys()

    def getNVar(self):
        # Return the number of variables
        return len(self.domains.keys())

    def isConsistent(self):
        # Check if the current assignments are consistent with the constraints
        consistent = True
        for k in self.constraints.keys():
            co = self.constraints[k]
            for c in co:
                if (self.assignments[k] is not None) and (self.assignments[c] is not None):
                    consistent = consistent and (self.assignments[k] != self.assignments[c])
        return consistent

    def getDomain(self, var):
        # Return the domain of a variable
        return self.domains[var]

    def assign(self, var, value):
        # Assign a value to a variable
        self.assignments[var] = value

    def unassign(self, var):
        # Unassigned a variable
        self.assignments[var] = None

    def getAssignments(self):
        # Return the current assignments
        return self.assignments

    def getNextAssignableVar(self):
        # Return the next unassigned variable with a non-empty domain
        for u in self.assignments.keys():
            if (self.assignments[u] is None) and self.domains[u] != []:
                return u

    def isSolved(self):
        # Check if the CSP is solved
        solved = True
        for a in self.assignments.values():
            solved = solved and (a is not None)
        return solved and self.isConsistent()


# Backtracking search algorithm
def backtrackcsp(csp, depth):
    if csp.isSolved():
        if csp.isConsistent():
            return csp
        return None

    next_var = csp.getNextAssignableVar()
    # If there are no more unassigned variables, return None
    if next_var is None:
        return None

    # For each value in the domain of the unassigned variable
    for value in csp.getDomain(next_var):
        csp.assign(next_var, value)
        result = backtrackcsp(csp, depth + 1)
        if result is not None:
            return result
        csp.unassign(next_var)

    return None


def main():
    # Main function to test the CSP and backtracking algorithm
    examplecsp = CSP(
        {'NWM': ['r', 'g', 'b'],
         'LP': ['r', 'g', 'b'],
         'LR': ['r', 'g', 'b'],
         'S': ['r', 'g', 'b'],
         'R': ['r', 'g', 'b'],
         'VR': ['r', 'g', 'b'],
         'VG': ['r', 'g', 'b'],
         'MSE': ['r', 'g', 'b']},
        {'NWM': ['S', 'LP', 'LR'],
         'LP': ['NWM', 'S', 'LR', 'MSE'],
         'LR': ['NWM', 'R', 'LP', 'MSE', 'VR'],
         'S': ['NWM', 'LP'],
         'R': ['LR'],
         'VR': ['LR', 'MSE', 'VG'],
         'VG': ['VR', 'MSE'],
         'MSE': ['LP', 'LR', 'VR', 'VG']})

    print(backtrackcsp(examplecsp, 0).getAssignments())


if __name__ == '__main__':
    main()