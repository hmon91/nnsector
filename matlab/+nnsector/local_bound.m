function sb = local_bound(net, y_lo, y_hi, varargin)
%LOCAL_BOUND  Local sector bound of a network about a point y*.
%   sb = nnsector.local_bound(net, y_lo, y_hi)
%   sb = nnsector.local_bound(net, y_lo, y_hi, 'ystar', ystar)
%   sb = nnsector.local_bound(..., 'Intervals', 'ibp')
%
%   Bounds the change pi(y) - pi(y*) of the network output for y in the box
%   y_lo <= y <= y_hi, which must contain y* (default y* = 0). The biases
%   cancel in the changes, so networks with biases are fine. It works for
%   every activation of nnsector.activation.
%
%   One input (p = 1): for every y ~= y* in the box, row by row,
%       sb.gamma1 <= (pi(y) - pi(y*)) / (y - y*) <= sb.gamma2.
%   y* = y_lo (y* at the lower corner of the box), any number of inputs:
%       sb.gamma1*(y - y*) <= pi(y) - pi(y*) <= sb.gamma2*(y - y*).
%   General: the box is cut at y* into pieces on which every y_k - y*_k
%   keeps one sign, given by sb.pieces(k).signs. On piece k,
%       sb.pieces(k).L*|y - y*| <= pi(y) - pi(y*) <= sb.pieces(k).U*|y - y*|.
%   With p > 1 inputs and y* ~= y_lo, gamma1 and gamma2 are empty: use pieces.
%
%   Options
%     'ystar'      the point y*, inside the box (default zeros(p,1))
%     'Intervals'  'intersect' (default): each preactivation interval is the
%                  intersection of the interval bound propagation (IBP)
%                  interval with the range of the linear bounds over the box;
%                  'ibp': the IBP interval alone
%
%   Other fields: sb.ystar, sb.pistar = pi(y*), sb.y_lo, sb.y_hi,
%   sb.intervals, sb.activation, and for each piece z_hi (the largest
%   |y - y*| on it) and case3 (neurons per hidden layer whose interval
%   contains nu*, Case 3, the loosest case).

    ip = inputParser;
    ip.FunctionName = 'nnsector.local_bound';
    ip.addParameter('ystar', [], @(v) isnumeric(v) && isreal(v));
    ip.addParameter('Intervals', 'intersect', ...
                    @(s) ischar(s) && any(strcmpi(s, {'intersect','ibp'})));
    ip.parse(varargin{:});

    p     = net.p;
    y_lo  = column(y_lo, p, 'y_lo');
    y_hi  = column(y_hi, p, 'y_hi');
    ystar = ip.Results.ystar;
    if isempty(ystar), ystar = 0; end
    ystar = column(ystar, p, 'ystar');
    if any(y_lo > y_hi)
        error('nnsector:local_bound:box', 'The box needs y_lo <= y_hi.');
    end
    if any(ystar < y_lo | ystar > y_hi)
        error('nnsector:local_bound:ystar', ...
              'ystar must lie in the box [y_lo, y_hi] (the default ystar is 0).');
    end
    intersect = strcmpi(ip.Results.Intervals, 'intersect');
    A = nnsector.activation(net.activation, net.alpha);

    % pieces of the box: on piece s, y = ystar + diag(s)*z with 0 <= z <= z_hi
    S = sign_patterns(y_lo, y_hi, ystar);
    if size(S,2) > 1024
        warning('nnsector:local_bound:pieces', ...
                'ystar cuts the box into %d pieces, which can be slow.', size(S,2));
    end
    pieces = struct('signs', {}, 'z_hi', {}, 'L', {}, 'U', {}, 'case3', {});
    for k = 1:size(S,2)
        s    = S(:,k);
        z_hi = (s > 0).*(y_hi - ystar) + (s < 0).*(ystar - y_lo);
        W = net.W;  b = net.b;
        b{1} = W{1}*ystar + b{1};       % the same network, seen from z
        W{1} = W{1}*diag(s);
        [L, U, ncase3] = incremental_bound(W, b, A, z_hi, intersect);
        pieces(k) = struct('signs', s, 'z_hi', z_hi, 'L', L, 'U', U, 'case3', ncase3);
    end

    gamma1 = [];  gamma2 = [];
    if p == 1                           % slopes of the secant through y*
        lo = zeros(net.m, numel(pieces));  hi = lo;
        for k = 1:numel(pieces)
            if pieces(k).signs > 0      % y >= y*: slopes in [L, U]
                lo(:,k) = pieces(k).L;   hi(:,k) = pieces(k).U;
            else                        % y <= y*: slopes in [-U, -L]
                lo(:,k) = -pieces(k).U;  hi(:,k) = -pieces(k).L;
            end
        end
        gamma1 = min(lo, [], 2);
        gamma2 = max(hi, [], 2);
    elseif numel(pieces) == 1 && all(pieces.signs > 0)
        gamma1 = pieces.L;
        gamma2 = pieces.U;
    end

    sb.gamma1     = gamma1;
    sb.gamma2     = gamma2;
    sb.ystar      = ystar;
    sb.pistar     = nnsector.forward(net, ystar);
    sb.y_lo       = y_lo;
    sb.y_hi       = y_hi;
    sb.pieces     = pieces;
    sb.intervals  = lower(ip.Results.Intervals);
    sb.activation = net.activation;
end

% -------------------------------------------------------------------------
function [L, U, ncase3] = incremental_bound(W, b, A, z_hi, intersect)
% The local bound on the box 0 <= z <= z_hi, about z* = 0:
%     L*z <= pi(z) - pi(0) <= U*z.
% Step S1 carries linear bounds on the changes nu~ = nu - nu* of the
% preactivations through the weights. Step S2 bounds act(nu) - act(nu*) on
% each neuron's interval by two slopes, in three cases: the interval lies
% above nu* (Case 1), below it (Case 2), or contains it (Case 3). The biases
% cancel in nu~ and enter only through the nominal preactivations nu*.
%
% The concrete intervals (interval bound propagation) are carried on the
% changes as well: d_lo <= nu~ <= d_hi. This is the same interval as
% nu* + [d_lo, d_hi], but every term of d_lo is <= 0 and every term of d_hi
% is >= 0. So when an interval ends at nu* (Cases 1 and 2), d_lo or d_hi is
% exactly 0, and rounding cannot move the neuron to Case 3.
    p    = size(W{1},2);
    nL   = numel(W);
    z_lo = zeros(p,1);

    % nominal forward pass at z = 0
    nu_s = cell(1,nL-1);
    w_s  = cell(1,nL);
    w_s{1} = z_lo;
    for i = 1:nL-1
        nu_s{i}  = W{i}*w_s{i} + b{i};
        w_s{i+1} = A.f(nu_s{i});
    end

    l     = eye(p);  u     = eye(p);    % the input changes by z
    dw_lo = z_lo;    dw_hi = z_hi;      % concrete interval on that change
    ncase3 = zeros(1, nL-1);

    for i = 1:nL
        Wp = max(W{i},0);  Wm = min(W{i},0);

        % S1: Lpre*z <= nu~ <= Upre*z
        Lpre = Wp*l + Wm*u;
        Upre = Wp*u + Wm*l;
        if i == nL                      % linear output layer: nu~ = pi(z) - pi(0)
            L = Lpre;  U = Upre;
            return
        end

        % concrete interval d_lo <= nu~ <= d_hi (interval bound propagation)
        [d_lo, d_hi] = box_range(W{i}, dw_lo, dw_hi);
        if intersect                    % nu~ also lies in [Lpre*z, Upre*z]
            [r_lo, ~] = box_range(Lpre, z_lo, z_hi);
            [~, r_hi] = box_range(Upre, z_lo, z_hi);
            d_lo = max(d_lo, r_lo);
            d_hi = min(d_hi, r_hi);
        end

        % S2: lower and upper slopes dL, dU of each neuron
        a = nu_s{i};
        nu_lo = a + d_lo;  nu_hi = a + d_hi;
        [s_inf, s_sup] = A.secant_range(a, nu_lo, nu_hi);
        case3 = d_lo < 0 & d_hi > 0;                        % straddles nu*
        case2 = d_lo < 0 & d_hi <= 0;                       % below nu*
        dL = s_inf;                dU = s_sup;              % Case 1: nu >= nu*
        dL(case2) = s_sup(case2);  dU(case2) = s_inf(case2);    % Case 2: nu <= nu*
        l  = diag(dL)*Lpre;             % bounds on the change of the layer output
        u  = diag(dU)*Upre;
        % Case 3: nu* is inside the interval. f never decreases, so
        % f(nu) - f(nu*) lies between k_neg*min(nu~, 0) and k_pos*max(nu~, 0),
        % with k_neg and k_pos the largest secant slopes left and right of
        % nu*. And z >= 0 gives min(nu~, 0) >= min(Lpre, 0)*z and
        % max(nu~, 0) <= max(Upre, 0)*z.
        [~, k_neg] = A.secant_range(a(case3), nu_lo(case3), a(case3));
        [~, k_pos] = A.secant_range(a(case3), a(case3), nu_hi(case3));
        l(case3,:) = k_neg.*min(Lpre(case3,:), 0);
        u(case3,:) = k_pos.*max(Upre(case3,:), 0);

        [w_lo, w_hi] = A.interval(nu_lo, nu_hi);
        dw_lo = min(w_lo - w_s{i+1}, 0);    % the interval always holds 0
        dw_hi = max(w_hi - w_s{i+1}, 0);
        ncase3(i) = nnz(case3);
    end
end

% -------------------------------------------------------------------------
function [lo, hi] = box_range(M, x_lo, x_hi)
% Componentwise range of M*x for x in the box [x_lo, x_hi].
    Mp = max(M,0);  Mm = min(M,0);
    lo = Mp*x_lo + Mm*x_hi;
    hi = Mp*x_hi + Mm*x_lo;
end

% -------------------------------------------------------------------------
function S = sign_patterns(y_lo, y_hi, ystar)
% Columns of S are the sign vectors of the pieces of the box cut at ystar.
% Coordinate k gets +1 if the box reaches above ystar_k, -1 if it reaches
% below, and +1 alone if the box is the single point ystar_k there.
    S = zeros(0,1);
    for k = 1:numel(ystar)
        sk = [];
        if y_hi(k) > ystar(k) || y_lo(k) == y_hi(k), sk(end+1) = 1;  end %#ok<AGROW>
        if y_lo(k) < ystar(k),                       sk(end+1) = -1; end %#ok<AGROW>
        S = [repmat(S, 1, numel(sk)); kron(sk, ones(1, size(S,2)))];
    end
end

% -------------------------------------------------------------------------
function v = column(v, p, name)
% v as a p x 1 column; a scalar is repeated.
    if ~isnumeric(v) || ~isreal(v) || any(~isfinite(v(:))) || ~(isscalar(v) || numel(v) == p)
        error('nnsector:local_bound:size', ...
              '%s must be a scalar or a vector with %d real entries.', name, p);
    end
    v = double(v(:)).*ones(p,1);
end
