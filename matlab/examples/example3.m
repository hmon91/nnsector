%% example3.m
%  The weights and biases of the demo network with each activation of
%  nnsector, about y* = 0 on [-1, 1]: the local bound, the exact range of
%  the secant slopes, and the global bound. The weights were trained for
%  tanh, so for the other activations this only shows how to use the
%  library.
% =========================================================================
clear; clc; close all;

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                       % the nnsector library folder

demo = nnsector.demo_network();
acts = {'tanh', 'sigmoid', 'relu', 'leakyrelu', 'softplus'};
y    = linspace(-1, 1, 200001);  y = y(y ~= 0); % grid for the exact slopes

fprintf('%-10s   %-18s   %-18s   %-18s   %s\n', 'activation', 'local bound', ...
        'exact slopes', 'global bound', 'check');
for k = 1:numel(acts)
    net = nnsector.network(demo.W, demo.b, 'Activation', acts{k});   % leaky ReLU: alpha = 0.01
    sb  = nnsector.local_bound(net, -1, 1);
    gb  = nnsector.global_bound(net);
    r   = (nnsector.forward(net, y) - sb.pistar) ./ y;      % (pi(y) - pi(0))/y
    res = nnsector.check(net, sb);
    fprintf('%-10s   [%7.4f, %7.4f]   [%7.4f, %7.4f]   [%7.4f, %7.4f]   %d\n', ...
            acts{k}, sb.gamma1, sb.gamma2, min(r), max(r), gb.Gamma1, gb.Gamma2, res.passed);
end
