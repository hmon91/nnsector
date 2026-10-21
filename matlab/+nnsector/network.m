function net = network(W, b, varargin)
%NETWORK  Feedforward neural network for nnsector.
%   net = nnsector.network(W)        network with zero biases
%   net = nnsector.network(W, b)     network with biases
%   net = nnsector.network(W, b, 'Activation', 'relu')
%   net = nnsector.network(W, b, 'Activation', 'leakyrelu', 'Alpha', 0.1)
%
%   W = {W1, ..., W_{q+1}} and b = {b1, ..., b_{q+1}} are the weights and
%   biases of a network with q hidden layers:
%       omega_0 = y,   omega_i = act(W_i*omega_{i-1} + b_i),  i = 1..q,
%       pi(y)   = W_{q+1}*omega_q + b_{q+1}.
%   W_i is l_i x l_{i-1} and b_i has l_i entries. The input size is p = l_0
%   and the output size is m = l_{q+1}. b = [] means zero biases.
%
%   All hidden layers use the same activation: 'tanh' (default), 'sigmoid',
%   'relu', 'leakyrelu' or 'softplus' (see nnsector.activation). 'Alpha' is
%   the slope of the leaky ReLU for x < 0 (default 0.01).
%
%   net is a struct with fields W, b, activation, alpha (leaky ReLU only,
%   otherwise empty), p, m and widths.

    ip = inputParser;
    ip.FunctionName = 'nnsector.network';
    ip.addParameter('Activation', 'tanh', @ischar);
    ip.addParameter('Alpha', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v)));
    ip.parse(varargin{:});
    if nargin < 2, b = []; end

    if ~iscell(W) || isempty(W)
        error('nnsector:network:weights', 'W must be a cell array {W1, ..., W_{q+1}}.');
    end
    W  = W(:).';
    nL = numel(W);
    for i = 1:nL
        Wi = W{i};
        if ~isnumeric(Wi) || ~isreal(Wi) || ~ismatrix(Wi) || isempty(Wi) || any(~isfinite(Wi(:)))
            error('nnsector:network:weights', 'W{%d} must be a nonempty real matrix.', i);
        end
        if i > 1 && size(Wi,2) ~= size(W{i-1},1)
            error('nnsector:network:size', ...
                  'W{%d} has %d columns, but W{%d} has %d rows.', ...
                  i, size(Wi,2), i-1, size(W{i-1},1));
        end
        W{i} = double(Wi);
    end

    if isempty(b)
        b = cellfun(@(Wi) zeros(size(Wi,1),1), W, 'UniformOutput', false);
    end
    if ~iscell(b) || numel(b) ~= nL
        error('nnsector:network:biases', 'b must be a cell array with %d entries, like W.', nL);
    end
    b = b(:).';
    for i = 1:nL
        bi = b{i};
        if isempty(bi), bi = zeros(size(W{i},1),1); end
        if ~isnumeric(bi) || ~isreal(bi) || numel(bi) ~= size(W{i},1) || any(~isfinite(bi(:)))
            error('nnsector:network:size', 'b{%d} must have %d real entries.', i, size(W{i},1));
        end
        b{i} = double(bi(:));
    end

    act   = lower(ip.Results.Activation);
    alpha = ip.Results.Alpha;           % [] means the default
    if strcmp(act, 'leakyrelu')
        if isempty(alpha), alpha = 0.01; end
        alpha = double(alpha);
    elseif ~isempty(alpha)
        warning('nnsector:network:alpha', ...
                'Alpha is used only by the ''leakyrelu'' activation; it is ignored here.');
        alpha = [];
    end
    nnsector.activation(act, alpha);    % errors if unsupported or alpha is bad

    net.W          = W;
    net.b          = b;
    net.activation = act;
    net.alpha      = alpha;
    net.p          = size(W{1},2);
    net.m          = size(W{end},1);
    net.widths     = [size(W{1},2), cellfun(@(Wi) size(Wi,1), W)];
end
