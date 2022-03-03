# To run
Run `stoichiometry.py` from the command line and enter chemical equations to balance.

# Input
`help` Prints out help message

`exit` / `quit` Exits the program

`{equation}` See help for description of equation syntax. Note: () cannot be nested within other () groups, and [] cannot be nested within other [] groups
because I wanted to implement with regex.

# Results
If there is only one possible ratio between the molecular quantities, displays the balanced equation.

If there are many possible ratios due to a combination of independent reactions, displays the equation
with dependent variables `a`, `b`, `c`, ... defined along with two or more independent variables.
