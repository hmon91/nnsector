function Y = forward(net, X)
%FORWARD  Evaluate the network.
%   Y = nnsector.forward(net, X) applies the network to every column of X
%   (p x N) and returns the outputs Y (m x N).

    if size(X,1) ~= net.p
        error('nnsector:forward:size', 'X must have %d rows (one column per input).', net.p);
    end
    A = nnsector.activation(net.activation, net.alpha);
    Z = X;
    for i = 1:numel(net.W)-1
        Z = A.f(net.W{i}*Z + net.b{i});
    end
    Y = net.W{end}*Z + net.b{end};
end
