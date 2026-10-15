function gb = global_bound(net)
%GLOBAL_BOUND  Global sector bound of a network.
%   gb = nnsector.global_bound(net) returns
%       gb.Gamma2 = c^q |W_{q+1}| ... |W_1|,   gb.Gamma1 = -gb.Gamma2,
%   where q is the number of hidden layers and c is the largest secant slope
%   of the activation: c = 1 for tanh, ReLU, leaky ReLU (alpha <= 1) and
%   softplus, and c = 1/4 for sigmoid.
%
%   With or without biases, for every y and y* (no box needed):
%       -Gamma2*|y - y*| <= pi(y) - pi(y*) <= Gamma2*|y - y*|,
%   so for one input, every secant slope (pi(y) - pi(y*))/(y - y*) lies in
%   [Gamma1, Gamma2]. This holds because every secant slope of the
%   activation lies in [-c, c], so each layer can grow a change by at most
%   c|W_i|.
%
%   With no biases and an activation with f(0) = 0 (tanh, ReLU, leaky ReLU),
%   pi(0) = 0, and for every y >= 0 this gives
%       Gamma1*y <= pi(y) <= Gamma2*y.
%
%   The product is accumulated from the input side, so the running product
%   keeps l0 columns: cost O(l0 * sum_i l_{i+1} l_i).

    A = nnsector.activation(net.activation, net.alpha);
    q = numel(net.W) - 1;
    c = max(abs(A.slope));
    P = abs(net.W{1});
    for i = 2:numel(net.W)
        P = abs(net.W{i})*P;
    end
    gb.Gamma2     = c^q * P;
    gb.Gamma1     = -gb.Gamma2;
    gb.c          = c;
    gb.activation = net.activation;
end
