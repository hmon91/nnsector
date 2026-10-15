function res = check(net, bound, varargin)
%CHECK  Test a sector bound on sample points.
%   res = nnsector.check(net, sb) tests a local bound from
%   nnsector.local_bound on random points of its box, points close to y*,
%   the box corners and, for one input, a fine grid.
%   res = nnsector.check(net, gb, 'Box', [y_lo y_hi]) tests a global bound
%   from nnsector.global_bound on a box (p x 2), about y* = 0 or 'ystar'.
%
%   res.max_violation  largest amount by which a sample breaks the bound
%                      (<= 0: every sample satisfies it)
%   res.passed         true if max_violation <= res.tol
%   res.tol            rounding tolerance
%   res.samples        number of sample points
%
%   Options: 'Samples' (random points, default 20000), 'Box', 'ystar'.
%   A sampling test cannot prove a bound; it can only catch a wrong one.
%   The state of the random number generator is restored afterwards.

    ip = inputParser;
    ip.FunctionName = 'nnsector.check';
    ip.addParameter('Samples', 20000, @(v) isnumeric(v) && isscalar(v) && v >= 1);
    ip.addParameter('Box', [], @isnumeric);
    ip.addParameter('ystar', [], @isnumeric);
    ip.parse(varargin{:});
    p = net.p;

    islocal = isfield(bound, 'pieces');
    if islocal
        y_lo = bound.y_lo;  y_hi = bound.y_hi;  ystar = bound.ystar;
    else
        box = ip.Results.Box;
        if ~isequal(size(box), [p 2])
            error('nnsector:check:box', ...
                  'Give the box for a global bound as ''Box'', [y_lo y_hi] (%d x 2).', p);
        end
        y_lo  = box(:,1);  y_hi = box(:,2);
        ystar = ip.Results.ystar;
        if isempty(ystar), ystar = zeros(p,1); end
        ystar = ystar(:).*ones(p,1);
    end

    st = rng;  rng(0);
    restore = onCleanup(@() rng(st));   %#ok<NASGU>

    w = y_hi - y_lo;
    Y = y_lo + w.*rand(p, round(ip.Results.Samples));     % random points
    Y = [Y, ystar + 1e-3*w.*(2*rand(p, 1000) - 1)];       % close to y*
    if p <= 12                                            % box corners
        Y = [Y, y_lo + w.*(dec2bin(0:2^p-1, p).' == '1')];
    end
    if p == 1                                             % fine grid
        Y = [Y, linspace(y_lo, y_hi, 100001)];
    end
    Y = min(max(Y, y_lo), y_hi);

    D  = nnsector.forward(net, Y) - nnsector.forward(net, ystar);   % pi(y) - pi(y*)
    Yt = Y - ystar;                                                  % y - y*
    v  = -inf;
    if islocal
        for k = 1:numel(bound.pieces)
            P  = bound.pieces(k);
            in = all(P.signs.*Yt >= 0, 1);
            Z  = abs(Yt(:,in));
            E  = [P.L*Z - D(:,in), D(:,in) - P.U*Z];
            v  = max([v; E(:)]);
        end
        if p == 1 && ~isempty(bound.gamma1)               % slope form
            far = abs(Yt) > 1e-4*max(w, eps);             % skip tiny steps
            R   = D(:,far)./Yt(far);
            E   = [bound.gamma1 - R, R - bound.gamma2];
            v   = max([v; E(:)]);
        end
    else
        Z = abs(Yt);
        E = [-bound.Gamma2*Z - D, D - bound.Gamma2*Z];
        v = max([v; E(:)]);
    end

    res.max_violation = v;
    res.tol     = 1e-9*(1 + max(abs(D(:))));
    res.passed  = v <= res.tol;
    res.samples = size(Y,2);
end
