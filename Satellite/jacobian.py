import sympy as sp
import numpy as np

def sym_jaco():
    x, y, z = sp.symbols('x, y, z')
    m = np.array([x**2 + y**2, x*y*z, sp.exp(x)])
    M = sp.Matrix(m)
    print(M.jacobian([x, y, z]))

if __name__ == "__main__":
    sym_jaco()