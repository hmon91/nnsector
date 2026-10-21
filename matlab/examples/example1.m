%% example1.m
%  Sector bounds of the demo network of nnsector: a small tanh network
%  trained to act like the controller u = -tanh(2y) - 0.5y (1 input, two
%  hidden layers of 8, 1 output, with biases). Shows a bound about y* = 0,
%  compares it with the exact secant slopes, and adds a bound about a point
%  y* inside the box. Run it from any folder.
% =========================================================================
clear; clc; close all;

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                       % the nnsector library folder

net = nnsector.demo_network();

%% 1. Local bound about y* = 0 on [-1, 1]
sb0 = nnsector.local_bound(net, -1, 1);
fprintf('Local bound, y* = 0 on [-1, 1]   : [%.4f, %.4f]\n', sb0.gamma1, sb0.gamma2);

y = linspace(-1, 1, 200001);  y = y(y ~= 0);
r = (nnsector.forward(net, y) - sb0.pistar) ./ y;       % (pi(y) - pi(0))/y
fprintf('  exact secant slopes             : [%.4f, %.4f]\n', min(r), max(r));

% the same with plain IBP intervals
sb_ibp = nnsector.local_bound(net, -1, 1, 'Intervals', 'ibp');
fprintf('  with plain IBP intervals        : [%.4f, %.4f]\n', sb_ibp.gamma1, sb_ibp.gamma2);

%% 2. Global bound
gb = nnsector.global_bound(net);
fprintf('Global bound                      : [%.4f, %.4f]\n', gb.Gamma1, gb.Gamma2);

%% 3. Local bound about y* = 0.5, inside the box [-1, 1]
% The box is cut at y* into [-1, 0.5] and [0.5, 1]; each piece gets its
% own bound, and gamma1, gamma2 hold for both:
%     gamma1 <= (pi(y) - pi(0.5)) / (y - 0.5) <= gamma2   for every y ~= 0.5.
sb1 = nnsector.local_bound(net, -1, 1, 'ystar', 0.5);
fprintf('Local bound, y* = 0.5 on [-1, 1] : [%.4f, %.4f]\n', sb1.gamma1, sb1.gamma2);
side = {'y <= y*', 'y >= y*'};
for k = 1:numel(sb1.pieces)
    P = sb1.pieces(k);
    fprintf('  piece %s : %.4f |y-y*| <= pi(y)-pi(y*) <= %.4f |y-y*|\n', ...
            side{(P.signs > 0) + 1}, P.L, P.U);
end

%% 4. Check the bounds on sample points
r0 = nnsector.check(net, sb0);
r1 = nnsector.check(net, sb1);
rg = nnsector.check(net, gb, 'Box', [-20 20]);
fprintf('Sample checks passed              : %d %d %d\n', r0.passed, r1.passed, rg.passed);

%% 5. Plot the network and the two local sectors
y = linspace(-1, 1, 400);
figure;  hold on;  box on;  grid on;
plot(y, nnsector.forward(net, y), 'k-', 'LineWidth', 1.5);
plot(y, sb0.pistar + sb0.gamma1*(y - sb0.ystar), 'r--', ...
     y, sb0.pistar + sb0.gamma2*(y - sb0.ystar), 'g--');
plot(y, sb1.pistar + sb1.gamma1*(y - sb1.ystar), 'r:', ...
     y, sb1.pistar + sb1.gamma2*(y - sb1.ystar), 'g:', 'LineWidth', 1.25);
plot(sb1.ystar, sb1.pistar, 'ko', 'MarkerFaceColor', 'k');
xlabel('y');  ylabel('u');
legend('\pi(y)', '\gamma_1 about y* = 0', '\gamma_2 about y* = 0', ...
       '\gamma_1 about y* = 0.5', '\gamma_2 about y* = 0.5', 'y* = 0.5', ...
       'Location', 'southwest');
