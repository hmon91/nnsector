%% example2.m
%  nnsector on networks with biases: a one-input network about a point y*
%  inside the box, and a two-input network, where the bound comes in pieces.
%  The networks are random, with a fixed seed.
% =========================================================================
clear; clc; close all;

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                       % the nnsector library folder
rng(1);

%% 1. One input, hidden layers of 10 and 5, y* = 0.5 in [-1, 2]
W   = {randn(10,1), 0.5*randn(5,10), randn(1,5)};
b   = {0.5*randn(10,1), 0.5*randn(5,1), 0.5*randn};
net = nnsector.network(W, b);

sb = nnsector.local_bound(net, -1, 2, 'ystar', 0.5);
fprintf('One input, y* = 0.5 on [-1, 2]\n');
fprintf('  pi(y*) = %.4f\n', sb.pistar);
fprintf('  gamma1 <= (pi(y)-pi(y*))/(y-y*) <= gamma2 :  [%.4f, %.4f]\n', sb.gamma1, sb.gamma2);
r = nnsector.check(net, sb);
fprintf('  sample check passed : %d\n\n', r.passed);

%% 2. Two inputs, two outputs, two hidden layers of 8, y* = (1, 0.5)
W2  = {0.7*randn(8,2), 0.5*randn(8,8), 0.5*randn(2,8)};
b2  = {0.3*randn(8,1), 0.3*randn(8,1), 0.3*randn(2,1)};
net2 = nnsector.network(W2, b2);

y_lo = [0; 0];  y_hi = [2; 1];  ystar = [1; 0.5];
sb2 = nnsector.local_bound(net2, y_lo, y_hi, 'ystar', ystar);
fprintf('Two inputs, y* = (1, 0.5) in [0,2] x [0,1]: %d pieces\n', numel(sb2.pieces));
for k = 1:numel(sb2.pieces)
    P = sb2.pieces(k);
    fprintf('  piece with signs (%+d, %+d):  L*|y-y*| <= pi(y)-pi(y*) <= U*|y-y*|\n', P.signs);
    fprintf('    L = %s\n    U = %s\n', mat2str(P.L, 4), mat2str(P.U, 4));
end
r2 = nnsector.check(net2, sb2);
fprintf('  sample check passed : %d\n\n', r2.passed);

% With y* at the lower corner there is one piece, and the bound has the
% form gamma1*(y - y*) <= pi(y) - pi(y*) <= gamma2*(y - y*).
sb3 = nnsector.local_bound(net2, y_lo, y_hi, 'ystar', y_lo);
fprintf('Two inputs, y* = y_lo: gamma1 = %s, gamma2 = %s\n', ...
        mat2str(sb3.gamma1, 4), mat2str(sb3.gamma2, 4));
