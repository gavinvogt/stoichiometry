'''
File: stoichiometry.py
Author: Gavin Vogt
This program uses basic linear algebra to solve stoichiometry problems.

Interesting resource covering the same topic:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1028.5627&rep=rep1&type=pdf
'''

# dependencies
import re
import numpy as np

HELP_STRING = """---- Solve Equation ----
Equation form:
    - Enter as a typical chemical equation
    - All atomic symbols must start with a single capital letter and
      may contain 0+ lowercase letters, followed by an optional number

Examples:
    - CO + O2 -> CO2
    - KNO3 -> KNO2 + O2
    - Na2S2O3 + AgBr -> NaBr + Na3AgS4O6

---- To Exit ----
Enter 'quit' or 'exit'"""

class EquationParseError(Exception):
    pass

def atomic_symbol_set(molecules):
    '''
    Gets the set of atomic symbols from a collection of Molecules
    '''
    atomic_symbols = set()
    for molecule in molecules:
        for atomic_symbol in molecule:
            atomic_symbols.add(atomic_symbol)
    return atomic_symbols

class ChemicalEquation:
    '''
    This class represents a chemical equation to balance

    Useful properties:
        - left: tuple of Molecules on left side of equation
        - right: tuple of Molecules on right side of equation

    Useful methods:
        - verify():
        - to_linalg(): converts to a matrix A in Ax=0 to solve
        - print_symbolic(): prints out the general symbolic solution to the equation
        - print_single(): prints out the single solution to the equation
    '''

    # Example: CH3NH2 + O2 -> CO2 + H2O + N2
    EQUATION_PATTERN = re.compile(r"^ \s* (.+) -> (.+) \s* $", re.VERBOSE)

    def __init__(self, equation_str):
        self._left, self._right = self._parse_equation(equation_str)
        self._all_atoms = tuple(atomic_symbol_set(self._left).union(atomic_symbol_set(self._right)))

    @property
    def left(self):
        '''
        Defines the property holding the Molecules on the left side of the equation
        '''
        return self._left

    @property
    def right(self):
        '''
        Defines the property holding the Molecules on the right side of the equation
        '''
        return self._right

    def verify(self):
        '''
        Verifies that the chemical equation could possibly be balanced
        Return: True if successfully verified, otherwise False
        '''
        # Verify the atomic symbols on each side
        left_atoms = atomic_symbol_set(self._left)
        right_atoms = atomic_symbol_set(self._right)
        if left_atoms != right_atoms:
            print("Error: unbalanced equation")
            right_missing = left_atoms - right_atoms
            if len(right_missing) > 0:
                print("Contained in left side only:", ", ".join(right_missing))

            left_missing = right_atoms - left_atoms
            if len(left_missing) > 0:
                print("Contained in right side only:", ", ".join(left_missing))
            return False

        return True

    def to_linalg(self):
        '''
        Converts the chemical equation to a linear algebra problem of the
        form Ax = b.
        x: column vector of coefficients for each molecule
        b: column vector of 0s
        Returns matrix A
        '''
        # Set up the matrix with correct dimensions
        left, right = self._left, self._right
        A = np.zeros((len(self._all_atoms), len(left) + len(right)), dtype='int')

        # Molecules on left side of equation (positive)
        for i in range(len(left)):
            molecule = left[i]
            j = 0
            for atomic_symbol in self._all_atoms:
                A[j, i] = molecule.get(atomic_symbol)
                j += 1

        # Molecules on right side of equation (negative)
        for i in range(len(right)):
            molecule = right[i]
            j = 0
            for atomic_symbol in self._all_atoms:
                A[j, len(left) + i] = -molecule.get(atomic_symbol)
                j += 1
        return A

    def print_symbolic(self):
        '''
        Prints out the chemical equation, inserting variable symbols for
        the quantities of each molecule
        '''
        # Print out system of equations
        left, right = self._left, self._right

        sections = []
        l = []
        for i in range(len(left)):
            l.append(f"{self.get_var(i)} {left[i].original}")
        sections.append(" + ".join(l))
        sections.append('->')
        l.clear()
        for i in range(len(right)):
            l.append(f"{self.get_var(len(left) + i)} {right[i].original}")
        sections.append(" + ".join(l))
        print(" ".join(sections))

    def print_single(self, values):
        '''
        Prints out the single solution to this chemical equation, using
        the values list of integers as the quantity for each molecule
        '''
        # Print out system of equations
        left, right = self._left, self._right

        sections = []
        l = []
        for i in range(len(left)):
            l.append(f"{values[i]} {left[i].original}")
        sections.append(" + ".join(l))
        sections.append('->')
        l.clear()
        for i in range(len(right)):
            l.append(f"{values[len(left) + i]} {right[i].original}")
        sections.append(" + ".join(l))
        print(" ".join(sections))

    LETTERS = 'abcdefghijklmnopqrstuvwxyz'

    def get_var(self, i):
        '''
        Gets name for variable in front of molecule `i` in the equation
        '''
        if len(self._all_atoms) <= 26:
            return self.LETTERS[i]
        else:
            return f"n{i}"

    def _parse_equation(self, equation: str):
        '''
        Parses the given stoichiometric equation
        Returns two tuples of Molecules representing the left / right sides of the equation.
        '''
        # Match equation to pattern
        m = re.match(self.EQUATION_PATTERN, equation)
        if m is None:
            raise EquationParseError('Failed to match equation pattern')

        # Parse the two sides of the equation
        left = self._parse_side(m.group(1))
        right = self._parse_side(m.group(2))
        return left, right

    # Example: CH3NH2 + O2
    def _parse_side(self, side: str):
        '''
        Parses a side of the stoichiometric equation.
        Returns a tuple of Molecule objects, where each Molecule contains its count
        and molecular formula.
        '''
        molecules = side.split("+")
        return tuple(Molecule(molecule) for molecule in molecules)

class Molecule:
    '''
    This class represents a molecule in a stoichiometric equation.
    For example, H2O has formula={'H': 2, 'O': 1}.

    Useful properties:
        - original: string representing the original molecular formula

    Useful methods:
        - get(): Get the number of a particular atom in the molecular formula
    '''

    # Example: CH3NH2
    MOLECULE_PATTERN = re.compile(r"^ \s* ([a-zA-Z0-9]+) \s* $", re.VERBOSE)

    # Example: CaCl2 -> Ca, Cl2
    ATOM_PATTERN = re.compile(r"([A-Z][a-z]*) (\d+)?", re.VERBOSE)

    def __init__(self, original):
        '''
        Constructs a Molecule, which contains the count and molecular formula
        '''
        # Sets _formula and _original
        self._parse_molecule(original)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._original})"

    def __iter__(self):
        '''
        Iterates over every atomic symbol in the formula
        '''
        return self._formula.__iter__()

    @property
    def original(self):
        '''
        Defines the property for the original molecular formula
        '''
        return self._original

    def get(self, atom: str):
        '''
        Gets the count of the given atom in the molecule.
        atom: str, representing the atomic symbol
        '''
        return self._formula.get(atom, 0)

    def _parse_molecule(self, molecule: str):
        '''
        Parses a molecule section in the equation. This includes the count (fixed number
        or _ to indicate needs solving) and the molecular formula
        '''
        # Match molecule to pattern
        m = re.match(self.MOLECULE_PATTERN, molecule)
        if m is None:
            raise EquationParseError('Failed to match molecule pattern')

        # Molecular equation
        self._parse_molecule_formula(m.group(1))

    def _parse_molecule_formula(self, formula: str):
        '''
        Parses a molecular formula and stores the result in a dict
        '''
        # Save the molecule information
        self._formula = {}
        self._original = formula

        # Load the counts for each atom
        m = re.search(self.ATOM_PATTERN, formula)
        while m is not None:
            # Verify that it starts at the beginning
            if m.start() != 0:
                raise EquationParseError(f"Failed to match molecular formula '{formula}'")

            # Get the current atom/count
            atom = m.group(1)
            n = 1 if m.group(2) is None else int(m.group(2))
            self._formula[atom] = self._formula.get(atom, 0) + n

            # Move on to the next atom/count
            formula = formula[m.end():]
            m = re.search(self.ATOM_PATTERN, formula)

        if len(formula) > 0:
            # Did not match the whole string
            raise EquationParseError(f"Failed to match molecular formula '{formula}'")

def get_lead(M, c):
    '''
    Finds a row with a nonzero "lead" value in column c in the matrix M.
    Swaps that row with row c. Ignores any rows < c, assuming they have
    been processed and should be left alone.
    '''
    row_count, column_count = M.shape
    for r in range(c, row_count):
        if M[r, c] != 0:
            # Found a row
            M[c], M[r] = M[r].copy(), M[c].copy()
            if M[c, c] < 0:
                # Negative; flip sign
                M[c] *= -1
            return M[c, c]

def rref(M):
    '''
    rref's the matrix M in-place.
    '''
    row_count, column_count = M.shape
    for c in range(min(row_count, column_count)):
        lead = get_lead(M, c)
        if lead is None:
            continue
        for r in range(row_count):
            if r != c and M[r, c] != 0:
                # Convert M[r] to have M[r, c] = 0
                M[r] = M[r] * lead - M[c] * M[r, c]

    # Reduce
    for r in range(row_count):
        gcd = np.gcd.reduce(M[r])
        if gcd != 0:
            M[r] //= gcd

def get_free_variables(A):
    '''
    Returns the list of free variables in matrix A.
    0 = first variable free, 1 = second free, ...
    '''
    # Find the free variables
    free_vars = []
    row_count, column_count = A.shape
    for c in range(column_count):
        # Guaranteed that every element A[c, i] = 0 for i < c due to rref
        if not (c < row_count and A[c, c] != 0):
            # Free variable
            free_vars.append(c)
    return free_vars

def multiple_free_variables(chemical_equation: ChemicalEquation, A):
    '''
    Displays solution to balanced equation for multiple free variables (severable
    viable ratios between molecules)
    chemical_equation: ChemicalEquation that was solved
    A: 2-D array of ints in the reduced matrix used to solve
    '''
    print("Solution:", end=" ")
    chemical_equation.print_symbolic()

    row_count, column_count = A.shape
    for r in range(row_count):
        # Guaranteed that every element A[r, i] = 0 for i < r due to rref
        if r < column_count and A[r, r] != 0:
            # Found a pivot
            dependencies = []
            for i in range(r + 1, column_count):
                if A[r, i] != 0:
                    coef = "" if A[r, i] == -1 else -A[r, i]
                    dependencies.append(f"{coef}{chemical_equation.get_var(i)}")

            # Print out the equation for this variable in the chemical equation
            coef = "" if A[r, r] == 1 else A[r, r]
            print(f"  [DEP] {coef}{chemical_equation.get_var(r)} =",
                " + ".join(dependencies))

    # Print out the free variables
    free_vars = get_free_variables(A)
    for free_var in free_vars:
        var = chemical_equation.get_var(free_var)
        print(f"  [IND] {var} = ?")

def single_free_variable(chemical_equation: ChemicalEquation, A, free_var: int):
    '''
    Displays solution to balanced equation for a single free variable (only one
    viable ratio between molecules)
    chemical_equation: ChemicalEquation that was solved
    A: 2-D array of ints in the reduced matrix used to solve
    free_var: int, representing the 0-index column number of the free variable
    '''
    # Find LCM of the leads
    leads = []
    row_count, column_count = A.shape
    for c in range(min(row_count, column_count)):
        if c != free_var:
            leads.append(A[c, c])
    lcm = np.lcm.reduce(leads)

    # Use the LCM as the value for the free variable
    values = []
    for r in range(max(row_count, column_count)):
        if r == free_var:
            values.append(lcm)
        elif A[r, free_var] != 0:
            values.append(-A[r, free_var] * (lcm // A[r, r]))

    print("Solution:", end=" ")
    chemical_equation.print_single(values)

def solve_equation(chemical_equation: ChemicalEquation):
    '''
    Solves the two sides of the equation
    '''
    if not chemical_equation.verify():
        # Verification failed
        return

    # Solve Ax = 0
    A = chemical_equation.to_linalg()
    rref(A)

    free_vars = get_free_variables(A)
    if len(free_vars) == 1:
        # Do lcm to get specific solution
        single_free_variable(chemical_equation, A, free_vars[0])
    else:
        # Multiple solutions available
        multiple_free_variables(chemical_equation, A)

def process_equation(equation: str):
    '''
    Parses and solves the given equation
    '''
    try:
        chemical_equation = ChemicalEquation(equation)
    except EquationParseError as e:
        print("Parsing error:", str(e))
        print("Enter 'help' for more information.")
    else:
        # Parsed equation successfully
        solve_equation(chemical_equation)

# Example inputs:
# CH3NH2 + O2 -> CO2 + H2O + N2
# O3 + NO -> NO2 + O2
# NaOH + H2SO4 -> Na2SO4 + H2O
# Na2S2O3 + AgBr -> NaBr + Na3AgS4O6
def main():
    while True:
        input_equation = input("Equation: ").strip()
        cmd = input_equation.lower()
        if cmd == 'help':
            # Help command
            print(HELP_STRING)
        elif cmd in ('exit', 'quit'):
            return
        else:
            # Assume they provided an equation
            process_equation(input_equation)
        print()

if __name__ == "__main__":
    main()
