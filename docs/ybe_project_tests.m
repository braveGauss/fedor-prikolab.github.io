%% Two-site XXZ / six-vertex experiment with symbolic + numeric tests

clear; clc;

%% Symbolic parameters (all real)
syms u v y real

%% Six-vertex weights (trigonometric XXZ parametrisation)
a  = sinh(u + y);   % a(u,y)
b  = sinh(u);       % b(u)
c  = sin(y);        % c(y) – anisotropy weight, independent of u
d  = -cosh(y);      % diagonal part of H, d(y) = -cos(i y) = -cosh(y)

%% Local spin-1/2 operators
s_plus  = [0 1; 0 0];
s_minus = [0 0; 1 0];
s_z     = [1 0; 0 -1];
I2      = eye(2);

%% Operators on V \otimes V \otimes V (threefold tensor product)

% Flip parts (s_+ \otimes s_- + s_- \otimes s_+) acting on different tensor slots
perm1 = kron(kron(s_plus, s_minus) + kron(s_minus, s_plus), I2);
perm2 = kron(s_plus, kron(I2, s_minus)) + kron(s_minus, kron(I2, s_plus));

% s_z ⊗ s_z parts in different slots
z1 = kron(kron(s_z, s_z), I2);
z2 = kron(kron(s_z, I2), s_z);

% Identity on V \otimes V \otimes V
I8 = kron(kron(I2, I2), I2);

%% Lax operators and monodromy (auxiliary \otimes 2-site quantum space)

L1 = 0.5*(a + b)*I8 + c*perm1 + 0.5*(a - b)*z1;
L2 = 0.5*(a + b)*I8 + c*perm2 + 0.5*(a - b)*z2;

% Two-site monodromy matrix
M = L2 * L1;  %#ok<NASGU>   % (not used further, but kept for completeness)

%% Explicit 2-site transfer matrix T(u) on V \otimes V (4×4)

a_u = a;      % = sinh(u + y)
b_u = b;      % = sinh(u)
c_y = c;      % = sin(y)

A = [a_u^2,         0,       0,      0;
     0,       a_u*b_u,  c_y^2,      0;
     0,            0,  a_u*b_u,     0;
     0,            0,       0,  b_u^2];

B = [b_u^2,         0,       0,      0;
     0,       a_u*b_u,      0,      0;
     0,        c_y^2,  a_u*b_u,     0;
     0,            0,       0,  a_u^2];

T = A + B;   % transfer matrix T(u) on a 2-site chain

%% Local 2-site Hamiltonian H (XXZ-type structure)

H = [ d,  0,  0,  0;
      0, -d,  2,  0;
      0,  2, -d,  0;
      0,  0,  0,  d];

%% Symbolic commutator [H, T(u)]

C_H = simplify(H*T - T*H, 'IgnoreAnalyticConstraints', true);

disp('Symbolic commutator [H, T(u)] =');
disp(C_H);

%% Second transfer matrix T(v) and commutator [T(u), T(v)]

% Weights with spectral parameter v
a_v = sinh(v + y);
b_v = sinh(v);
% c_y = sin(y) is the same anisotropy weight

A_new = [a_v^2,         0,       0,      0;
         0,       a_v*b_v,  c_y^2,      0;
         0,            0,  a_v*b_v,     0;
         0,            0,       0,  b_v^2];

B_new = [b_v^2,         0,       0,      0;
         0,       a_v*b_v,      0,      0;
         0,        c_y^2,  a_v*b_v,     0;
         0,            0,       0,  a_v^2];

T_new = A_new + B_new;   % transfer matrix T(v)

C_T = simplify(T_new*T - T*T_new, 'IgnoreAnalyticConstraints', true);

disp('Symbolic commutator [T(v), T(u)] =');
disp(C_T);

%% Numeric tests for seed values

% Choose some seed values (all real, avoid sin(y) = 0 to stay generic)
u_vals = [0.3, 0.7];
v_vals = [1.1, 1.5];
y_vals = [0.4, 0.8];   % you can reuse same y or vary; here we vary with index

fprintf('\n===== NUMERIC TESTS =====\n');
for k = 1:numel(u_vals)
    u0 = u_vals(k);
    v0 = v_vals(k);
    y0 = y_vals(k);

    fprintf('\nTest #%d: u = %.3f, v = %.3f, y = %.3f\n', k, u0, v0, y0);

    % Substitute and evaluate numeric commutators
    CH_num = double(subs(C_H, {u, y}, {u0, y0}));
    CT_num = double(subs(C_T, {u, v, y}, {u0, v0, y0}));

    % Compute norms / max absolute entry for diagnostics
    norm_CH = max(abs(CH_num(:)));
    norm_CT = max(abs(CT_num(:)));

    fprintf('  max |[H, T(u)]|     ≈ %.3e\n', norm_CH);
    fprintf('  max |[T(v), T(u)]| ≈ %.3e\n', norm_CT);

    % Optionally, display the numeric matrices if you want to inspect them:
    % disp('  [H, T(u)] numerically:');
    % disp(CH_num);
    % disp('  [T(v), T(u)] numerically:');
    % disp(CT_num);
end

fprintf('\nDone.\n');

