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
    - Can perform grouping with () and []
    - Can add charge with {x+} or {x-}
    
Examples:
    - CO + O2 -> CO2
    - KNO3 -> KNO2 + O2
    - Na2S2O3 + AgBr -> NaBr + Na3AgS4O6
    - Mg(OH)2 + H3PO4 -> H2O + Mg3(PO4)2
    - Fe{2+} + Cr2O7{2-} + H{+} -> Cr{3+} + H2O + Fe{3+}

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
    
    # Finds anything within curly brackets
    CURLY_BRACKETS_PATTERN = re.compile(r" ( { [^}]* [+-]+ [^{]* } ) ", re.VERBOSE)
    
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
        
        # Verify the charges
        return self._verify_charges()
    
    def _verify_charges(self):
        '''
        Verifies that the charges on either side can be balanced
        '''
        # Check for positive/negative on left
        left_has_positive = False
        left_has_negative = False
        for molecule in self._left:
            # Check positive or negative
            if molecule.charge > 0:
                left_has_positive = True
            elif molecule.charge < 0:
                left_has_negative = True
        
        # Check for positive/negative on right
        right_has_positive = False
        right_has_negative = False
        for molecule in self._right:
            # Check positive or negative
            if molecule.charge > 0:
                right_has_positive = True
            elif molecule.charge < 0:
                right_has_negative = True
        
        # Check the balance
        if left_has_positive and not (right_has_positive or left_has_negative):
            # Positive on left with no way to balance
            print("Unbalanced {+} on left side")
            return False
        elif left_has_negative and not (right_has_negative or left_has_positive):
            # Negative on left with no way to balance
            print("Unbalanced {-} on left side")
            return False
        elif right_has_positive and not (left_has_positive or right_has_negative):
            # Positive on right with no way to balance
            print("Unbalanced {+} on right side")
            return False
        elif right_has_negative and not (left_has_negative or right_has_positive):
            # Negative on right with no way to balance
            print("Unbalanced {-} on right side")
            return False
        else:
            # Charges can be balanced
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
        A = np.zeros((len(self._all_atoms) + 1, len(left) + len(right)), dtype='int')
        
        # Molecules on left side of equation (positive)
        for i in range(len(left)):
            molecule = left[i]
            j = 0
            for atomic_symbol in self._all_atoms:
                A[j, i] = molecule.get(atomic_symbol)
                j += 1
            # Add the charge to bottom row
            A[j, i] = molecule.charge
        
        # Molecules on right side of equation (negative)
        for i in range(len(right)):
            molecule = right[i]
            j = 0
            for atomic_symbol in self._all_atoms:
                A[j, len(left) + i] = -molecule.get(atomic_symbol)
                j += 1
            # Add the (negative) charge to bottom row
            A[j, len(left) + i] = -molecule.charge
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
            raise EquationParseError("Equation must match '<left> -> <right>' pattern")
        
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
        # Need to avoid splitting on the + inside {charge}
        pluses = [m.start() for m in re.finditer(r"\+", side)]
        molecules = []
        
        # Each curly range is [start, end] of curly braces to ignore
        curly_ranges = [(m.start(), m.end() - 1) for m in re.finditer(self.CURLY_BRACKETS_PATTERN, side)]
        i = 0
        for plus_index in pluses:
            # Make sure not in any of the curly ranges
            valid_for_split = True
            for start, end in curly_ranges:
                if start <= plus_index <= end:
                    # Plus is inside the {}; ignore
                    valid_for_split = False
                    break
            if valid_for_split:
                # Was not in a curly range; this is a valid slice
                molecules.append(side[i: plus_index])
                i = plus_index + 1
        if i < len(side):
            molecules.append(side[i:])
        
        # At this point, the molecules list has been properly split up on the non-{} plus signs
        return tuple(Molecule.parse(molecule) for molecule in molecules)


class Molecule:
    '''
    This class represents a molecule in a stoichiometric equation.
    For example, H2O has formula={'H': 2, 'O': 1}.
    
    Useful properties:
        - original: string representing the original molecular formula
        - charge: charge of the molecule
    
    Useful methods:
        - get(): Get the number of a particular atom in the molecular formula
    '''
    
    # Example: Cl2
    # With charge: Fe3{2+}
    ATOM_PATTERN = re.compile(r"([A-Z][a-z]*) (\d+)? ({.+})?", re.VERBOSE)
    
    # Example: (CO)3
    GROUP_PATTERN1 = re.compile(r"^ \( \s* ([^()]+) \s* \) (\d+)?", re.VERBOSE)
    # Example: [CO{2-}]
    GROUP_PATTERN2 = re.compile(r"^ \[ \s* ([^[\]]+) \s* \] (\d+)?", re.VERBOSE)
    
    # Example: O{2-} (must have integer since pattern2 will handle {-} and {+})
    CHARGE_PATTERN1 = re.compile(r"^ { \s* (\d+) \s* ([+-]) \s* } $", re.VERBOSE)
    # Example: O{-2}, {+}
    CHARGE_PATTERN2 = re.compile(r"^ { \s* ([+-]) \s* (\d+)? \s* } $", re.VERBOSE)
    
    def __init__(self):
        '''
        Constructs an empty Molecule.
        Do not construct manually; use Molecule.parse() instead
        '''
        # Empty Molecule
        self._formula = {}
        self._charge = 0
        self._original = ""
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self._original})"
    
    def __iter__(self):
        '''
        Iterates over every atomic symbol in the formula
        '''
        return self._formula.__iter__()
    
    def __add__(self, other):
        # Can only add Molecules
        if not isinstance(other, self.__class__):
            return NotImplemented
        
        # Create empty Molecule
        new_molecule = Molecule()
        
        # Add original formulas together (may be inaccurate)
        new_molecule._original = self._original + other._original
        
        # Add the atoms and charges together
        new_molecule._formula = self._formula.copy()
        for atom in other:
            new_molecule._formula[atom] = new_molecule.get(atom) + other.get(atom)
        new_molecule._charge = self._charge + other._charge
        return new_molecule
    
    def __mul__(self, count: int):
        # Only multiply by integers
        if type(count) != int:
            return NotImplemented
        
        # Create empty Molecule
        new_molecule = Molecule()
        for atom in self._formula:
            new_molecule._formula[atom] = self._formula[atom] * count
        new_molecule._charge = self._charge * count
        return new_molecule
    
    @property
    def original(self):
        '''
        Defines the property for the original molecular formula
        '''
        return self._original
    
    @property
    def charge(self):
        '''
        Defines the property for the charge of the molecule
        '''
        return self._charge
    
    def get(self, atom: str):
        '''
        Gets the count of the given atom in the molecule.
        atom: str, representing the atomic symbol
        '''
        return self._formula.get(atom, 0)
    
    def print_info(self):
        '''
        Helpful debug method for printing out information about the Molecule
        '''
        print("Molecule:", self._original)
        for atom, count in self._formula.items():
            print(f"  {atom}: {count}")
        print("  Charge:", self._charge)
    
    @classmethod
    def parse(cls, molecule_str: str):
        '''
        Parses the string representing the molecular formula and returns
        the associated Molecule
        '''
        # Strip any whitespace off ends
        molecule_str = molecule_str.strip()
        if molecule_str == "":
            # Empty molecule
            return cls()
        
        # Check if () or [] grouping pattern
        m = re.match(cls.GROUP_PATTERN1, molecule_str)
        if m is None:
            # Try with [] brackets instead
            m = re.match(cls.GROUP_PATTERN2, molecule_str)
        if m is not None:
            # Either matches () or [] grouping
            # (<molecule>)<int> | (<molecule>)
            # [<molecule>)<int> | [<molecule>]
            print("parsing group", m.group(1))
            count = 1 if m.group(2) is None else int(m.group(2))
            molecule = cls.parse(m.group(1)) * count + cls.parse(molecule_str[m.end():])
        else:
            # Not grouping pattern
            # <atoms><molecule> | <atoms>
            molecule = cls._parse_formula(molecule_str)
        
        # Set the original string
        molecule._original = molecule_str
        return molecule
    
    @classmethod
    def _parse_formula(cls, molecule_str: str):
        # Read the first segment of atoms
        m = re.search(cls.ATOM_PATTERN, molecule_str)
        if m is None:
            raise EquationParseError(f"Invalid atoms in '{molecule_str}'")
        elif m.start() != 0:
            # must start parsing from start of input
            raise EquationParseError(f"Invalid start sequence found in '{molecule_str}'")
        
        # Create Molecule out of these atoms
        atomic_symbol, count, charge_str = m.groups()
        if atomic_symbol is None and count is not None:
            # Tried to give count but no atoms
            raise EquationParseError(f"Must provide atomic symbol in '{molecule_str}'")
        elif atomic_symbol is None and charge_str is None:
            # Has neither atoms nor free charges
            raise EquationParseError(f"Must provide atom(s) or charge in '{molecule_str}'")
        
        # Valid symbol, count, and charge string
        molecule = cls()
        if atomic_symbol is not None:
            # Molecule with `count` number of `atomic_symbol` atoms
            count = 1 if count is None else int(count)
            molecule._formula[atomic_symbol] = count
        if charge_str is not None:
            # Parse charge
            molecule._charge = cls._parse_charge(charge_str)
        
        # Return created Molecule and parse the rest of the line
        return molecule + cls.parse(molecule_str[m.end():])
    
    @classmethod
    def _parse_charge(cls, charge_str: str):
        '''
        Parses the charge in the given charge_str.
        Examples: {2-} {-2} {+3} {3+} {+} {-}
        Return: int of charge amount (negative or positive)
        '''
        m = re.match(cls.CHARGE_PATTERN1, charge_str)
        if m is None:
            # Try other pattern
            m = re.match(cls.CHARGE_PATTERN2, charge_str)
            if m is None:
                # Both charge patterns failed
                raise EquationParseError(f"Failed to match charge '{charge_str}'")
            else:
                # O{-2} form (integer optional)
                sign, charge = m.groups()
        else:
            # O{2-} form (must include integer)
            charge, sign = m.groups()
        
        charge = 1 if charge is None else int(charge)
        if sign == "-":
            # Charge was negative
            return -charge
        else:
            # Charge was positive
            return charge

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
            dependencies_str = " + ".join(dependencies)
            
            if A[r, r] != 1:
                pivot_coef = A[r, r]
                if len(dependencies) == 1:
                    dependencies_str + f"{dependencies_str} / {pivot_coef}"
                else:
                    dependencies_str = f"({dependencies_str}) / {pivot_coef}"
            
            # Print out the equation for this variable in the chemical equation
            print(f"  [DEP] {chemical_equation.get_var(r)} =", dependencies_str)
    
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
    if len(free_vars) == 0:
        # No solution found
        print("No solution found")
    elif len(free_vars) == 1:
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
# Mg(OH)2 + H3PO4 -> H2O + Mg3(PO4)2
# Fe{2+} + Cr2O7{2-} + H{+} -> Cr{3+} + H2O + Fe{3+}
def main():
    while True:
        input_equation = input("Equation: ").strip()
        cmd = input_equation.lower()
        if len(cmd) == 0:
            # Ignore empty input
            continue
        elif cmd == 'help':
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

