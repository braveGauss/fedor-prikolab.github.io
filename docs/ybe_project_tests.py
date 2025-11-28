# %% [markdown]
# # Two-site XXZ / six-vertex experiment (SymPy + numeric tests)

# %% [markdown]
# ### 1. Imports and symbolic setup

import sympy as sp
import numpy as np

# Symbolic parameters (all real)
u, v, y = sp.symbols('u v y', real=True)

# Six-vertex / XXZ weights
a = sp.sinh(u + y)   # a(u, y)
b = sp.sinh(u)       # b(u)
c = sp.sin(y)        # c(y), anisotropy weight
d = -sp.cosh(y)      # d(y) = -cos(i y) = -cosh(y), diagonal part of H

# Local spin-1/2 operators (2x2 matrices)
s_plus  = sp.Matrix([[0, 1],
                     [0, 0]])
s_minus = sp.Matrix([[0, 0],
                     [1, 0]])
s_z     = sp.Matrix([[1, 0],
                     [0,-1]])
I2      = sp.eye(2)

# %% [markdown]
# ### 2. Operators on \(V \otimes V \otimes V\), Lax operators, and monodromy

# Flip parts acting on different tensor slots
perm1 = sp.kronecker_product(
    sp.kronecker_product(s_plus, s_minus) + sp.kronecker_product(s_minus, s_plus),
    I2
)
perm2 = (sp.kronecker_product(s_plus, sp.kronecker_product(I2, s_minus)) +
         sp.kronecker_product(s_minus, sp.kronecker_product(I2, s_plus)))

# s_z ⊗ s_z parts in different slots
z1 = sp.kronecker_product(sp.kronecker_product(s_z, s_z), I2)
z2 = sp.kronecker_product(sp.kronecker_product(s_z, I2), s_z)

# Identity on V \times V \otimes V
I8 = sp.eye(8)

# Lax operators L1, L2 (8x8) and monodromy M = L2 * L1
L1 = sp.Rational(1, 2) * (a + b) * I8 + c * perm1 + sp.Rational(1, 2) * (a - b) * z1
L2 = sp.Rational(1, 2) * (a + b) * I8 + c * perm2 + sp.Rational(1, 2) * (a - b) * z2

M = L2 * L1  # monodromy (not used later, but kept for completeness)

# %% [markdown]
# ### 3. Two-site transfer matrix \(T(u)\), Hamiltonian \(H\), and commutator \([H, T(u)]\)

# Explicit 4x4 transfer matrix T(u) on V ⊗ V
a_u = a      # = sinh(u + y)
b_u = b      # = sinh(u)
c_y = c      # = sin(y)

A = sp.Matrix([
    [a_u**2,      0,       0,      0],
    [0,      a_u*b_u,  c_y**2,     0],
    [0,           0,  a_u*b_u,     0],
    [0,           0,       0,  b_u**2]
])

B = sp.Matrix([
    [b_u**2,      0,       0,      0],
    [0,      a_u*b_u,     0,       0],
    [0,       c_y**2, a_u*b_u,     0],
    [0,           0,       0,  a_u**2]
])

T = A + B  # transfer matrix T(u)

# Local 2-site Hamiltonian H (XXZ-type)
H = sp.Matrix([
    [ d,  0,  0,  0],
    [ 0, -d,  2,  0],
    [ 0,  2, -d,  0],
    [ 0,  0,  0,  d]
])

# Symbolic commutator [H, T(u)]
C_H = sp.simplify(H * T - T * H)

print("Symbolic commutator [H, T(u)] =")
sp.pprint(C_H)
print()

# %% [markdown]
# ### 4. Second transfer matrix \(T(v)\) and commutator \([T(u), T(v)]\)

# Weights with spectral parameter v
a_v = sp.sinh(v + y)
b_v = sp.sinh(v)
# c_y = sin(y) is the same anisotropy weight

A_new = sp.Matrix([
    [a_v**2,      0,       0,      0],
    [0,      a_v*b_v, c_y**2,      0],
    [0,           0,  a_v*b_v,     0],
    [0,           0,       0,  b_v**2]
])

B_new = sp.Matrix([
    [b_v**2,      0,       0,      0],
    [0,      a_v*b_v,     0,       0],
    [0,       c_y**2, a_v*b_v,     0],
    [0,           0,       0,  a_v**2]
])

T_new = A_new + B_new  # transfer matrix T(v)

# Symbolic commutator [T(v), T(u)]
C_T = sp.simplify(T_new * T - T * T_new)

print("Symbolic commutator [T(v), T(u)] =")
sp.pprint(C_T)
print()

# %% [markdown]
# ### 5. Numeric tests for some seed values of (u, v, y)

# Choose some seed values (all real; avoid sin(y) = 0 to stay generic)
u_vals = [0.3, 0.7]
v_vals = [1.1, 1.5]
y_vals = [0.4, 0.8]

print("===== NUMERIC TESTS =====")
for k, (u0, v0, y0) in enumerate(zip(u_vals, v_vals, y_vals), start=1):
    print(f"\nTest #{k}: u = {u0:.3f}, v = {v0:.3f}, y = {y0:.3f}")

    # Substitute and evaluate numeric commutators
    CH_num = sp.Matrix(C_H.subs({u: u0, y: y0})).evalf()
    CT_num = sp.Matrix(C_T.subs({u: u0, v: v0, y: y0})).evalf()

    # Convert to numpy arrays to compute norms easily
    CH_np = np.array(CH_num.tolist(), dtype=np.complex128)
    CT_np = np.array(CT_num.tolist(), dtype=np.complex128)

    norm_CH = np.max(np.abs(CH_np))
    norm_CT = np.max(np.abs(CT_np))

    print(f"  max |[H, T(u)]|     ≈ {norm_CH:.3e}")
    print(f"  max |[T(v), T(u)]| ≈ {norm_CT:.3e}")

    # If you want, you can also inspect the actual matrices:
    # print("  [H, T(u)] numerically =")
    # sp.pprint(CH_num)
    # print("  [T(v), T(u)] numerically =")
    # sp.pprint(CT_num)

print("\nDone.")
