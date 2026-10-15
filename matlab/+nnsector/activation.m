function A = activation(name, alpha)
%ACTIVATION  Activation functions known to nnsector.
%   A = nnsector.activation(NAME) returns what the bound functions need to
%   know about the activation NAME:
%       'tanh'        tanh(x)
%       'sigmoid'     1/(1 + exp(-x))
%       'relu'        max(x, 0)
%       'leakyrelu'   max(x, 0) + alpha*min(x, 0),  alpha >= 0
%       'softplus'    log(1 + exp(x))
%   A = nnsector.activation('leakyrelu', ALPHA) sets alpha (default 0.01).
%   None of them ever decreases, which the local bound relies on.
%
%     A.name          name of the activation
%     A.f             the activation, applied elementwise
%     A.interval      [lo2, hi2] = A.interval(lo, hi): range of A.f over [lo, hi]
%     A.slope         [smin smax]: every secant slope (f(x)-f(a))/(x-a) lies in
%                     this interval (used by nnsector.global_bound)
%     A.secant_range  [s_inf, s_sup] = A.secant_range(a, x_lo, x_hi): smallest
%                     and largest secant slope (f(x)-f(a))/(x-a) for x ~= a in
%                     [x_lo, x_hi]. At an end point equal to a, the slope there
%                     is the one-sided derivative from inside the interval
%                     (used by nnsector.local_bound)
%
%   a, x_lo and x_hi are column vectors with one entry per neuron.
%
%   To add an activation that never decreases: add a case below and its
%   secant_range function.

    if nargin < 2 || isempty(alpha), alpha = 0.01; end

    switch lower(name)
        case 'tanh'
            A.f            = @tanh;
            A.slope        = [0 1];
            A.secant_range = @tanh_secant_range;
        case 'sigmoid'
            A.f            = @(x) 1./(1 + exp(-x));
            A.slope        = [0 0.25];
            A.secant_range = @sigmoid_secant_range;
        case 'relu'
            A.f            = @(x) max(x, 0);
            A.slope        = [0 1];
            A.secant_range = @(a, x_lo, x_hi) leaky_secant_range(a, x_lo, x_hi, 0);
        case 'leakyrelu'
            if ~(isnumeric(alpha) && isscalar(alpha) && isreal(alpha) && ...
                 isfinite(alpha) && alpha >= 0)
                error('nnsector:activation:alpha', ...
                      'The leaky ReLU slope alpha must be a real number >= 0.');
            end
            alpha          = double(alpha);
            A.f            = @(x) max(x, 0) + alpha*min(x, 0);
            A.slope        = [min(alpha, 1) max(alpha, 1)];
            A.secant_range = @(a, x_lo, x_hi) leaky_secant_range(a, x_lo, x_hi, alpha);
        case 'softplus'
            A.f            = @softplus;
            A.slope        = [0 1];
            A.secant_range = @softplus_secant_range;
        otherwise
            error('nnsector:activation:unsupported', ...
                  ['Activation ''%s'' is not supported. Supported: ''tanh'', ' ...
                   '''sigmoid'', ''relu'', ''leakyrelu'', ''softplus''.'], name);
    end
    A.name     = lower(name);
    f          = A.f;
    A.interval = @(lo, hi) deal(f(lo), f(hi));     % every f here never decreases
end

% ---- tanh ------------------------------------------------------------------
function [s_inf, s_sup] = tanh_secant_range(a, x_lo, x_hi)
% Smallest and largest secant slope s(x) = (tanh(x) - tanh(a))/(x - a) of
% tanh about a, for x in [x_lo, x_hi].
% tanh is convex for x < 0 and concave for x > 0, so s increases up to a
% single peak x_c and decreases after it. Hence the infimum is at an end
% point, and the supremum is at an end point or at x_c.
    s_lo  = tanh_secant(x_lo, a);
    s_hi  = tanh_secant(x_hi, a);
    s_inf = min(s_lo, s_hi);
    s_sup = max(s_lo, s_hi);
    xc    = tanh_secant_peak(a);
    in    = x_lo <= xc & xc <= x_hi;
    s_sup(in) = tanh_secant(xc(in), a(in));
end

% -------------------------------------------------------------------------
function s = tanh_secant(x, a)
% s = (tanh(x) - tanh(a))/(x - a), and s = sech(a)^2 at x = a. Near x = a
% the identity tanh(x) - tanh(a) = sinh(x-a)/(cosh(x)cosh(a)) avoids the
% cancellation in the difference quotient.
    d  = x - a;
    s  = (tanh(x) - tanh(a))./d;
    k  = abs(d) < 1;
    r  = ones(size(d));
    nz = d ~= 0;
    r(nz) = sinh(d(nz))./d(nz);         % sinh(d)/d, equal to 1 at d = 0
    s(k)  = r(k)./(cosh(x(k)).*cosh(a(k)));
end

% -------------------------------------------------------------------------
function xc = tanh_secant_peak(a)
% Peak x_c of the secant slope about a: the point where the tangent of tanh
% passes through (a, tanh(a)), the root of
%     g(x) = sech(x)^2 (x - a) - (tanh(x) - tanh(a)).
% x_c = 0 for a = 0, x_c has the sign opposite to a, and x_c(-a) = -x_c(a).
% The root is found for -|a| on x >= 0, where g falls from |a| - tanh|a| >= 0
% to -(1 + tanh|a|) < 0, by bisection on all neurons at once.
    b  = abs(a);
    g  = @(x) sech(x).^2.*(x + b) - tanh(x) - tanh(b);
    lo = zeros(size(b));
    hi = ones(size(b));
    while any(g(hi) > 0)
        k = g(hi) > 0;
        hi(k) = 2*hi(k);
    end
    for it = 1:60
        mid = (lo + hi)/2;
        k   = g(mid) > 0;
        lo(k)  = mid(k);
        hi(~k) = mid(~k);
    end
    xc = -sign(a).*(lo + hi)/2;
end

% ---- sigmoid ---------------------------------------------------------------
function [s_inf, s_sup] = sigmoid_secant_range(a, x_lo, x_hi)
% sigmoid(x) = (1 + tanh(x/2))/2, so its secant slope about a at x is a
% quarter of the secant slope of tanh about a/2 at x/2.
    [s_inf, s_sup] = tanh_secant_range(a/2, x_lo/2, x_hi/2);
    s_inf = s_inf/4;
    s_sup = s_sup/4;
end

% ---- ReLU and leaky ReLU ---------------------------------------------------
function [s_inf, s_sup] = leaky_secant_range(a, x_lo, x_hi, alpha)
% f(x) = max(x,0) + alpha*min(x,0) is convex for alpha <= 1 and concave for
% alpha >= 1, so its secant slope about a only rises (or only falls) with x,
% and the extremes are at the end points. At an end point equal to a, the
% slope is the one-sided derivative from inside the interval: from the right
% at x_lo, from the left at x_hi. ReLU is alpha = 0.
    s_lo = leaky_secant(x_lo, a, alpha);
    s_hi = leaky_secant(x_hi, a, alpha);
    k = x_lo == a;  s_lo(k) = alpha + (1 - alpha)*(a(k) >= 0);   % right derivative
    k = x_hi == a;  s_hi(k) = alpha + (1 - alpha)*(a(k) > 0);    % left derivative
    s_inf = min(s_lo, s_hi);
    s_sup = max(s_lo, s_hi);
end

% -------------------------------------------------------------------------
function s = leaky_secant(x, a, alpha)
% Secant slope (f(x) - f(a))/(x - a) of the leaky ReLU for x ~= a, written
% so that no difference of nearly equal numbers is formed.
    s = ones(size(x));                                % x and a both >= 0
    k = x <= 0 & a <= 0;  s(k) = alpha;               % x and a both <= 0
    k = a > 0 & x < 0;    s(k) = (a(k) - alpha*x(k))./(a(k) - x(k));
    k = a < 0 & x > 0;    s(k) = (x(k) - alpha*a(k))./(x(k) - a(k));
end

% ---- softplus --------------------------------------------------------------
function y = softplus(x)
% log(1 + exp(x)) without overflow for large x.
    y = max(x, 0) + log1p(exp(-abs(x)));
end

% -------------------------------------------------------------------------
function [s_inf, s_sup] = softplus_secant_range(a, x_lo, x_hi)
% softplus is convex, so its secant slope about a rises with x, and the
% extremes are at the end points.
    s_lo  = softplus_secant(x_lo, a);
    s_hi  = softplus_secant(x_hi, a);
    s_inf = min(s_lo, s_hi);
    s_sup = max(s_lo, s_hi);
end

% -------------------------------------------------------------------------
function s = softplus_secant(x, a)
% s = (softplus(x) - softplus(a))/(x - a), and s = sigmoid(a) at x = a. Near
% x = a, softplus(x) - softplus(a) = log1p(sigmoid(a)*expm1(x - a)) avoids the
% cancellation in the difference quotient.
    d  = x - a;
    s  = (softplus(x) - softplus(a))./d;
    k  = abs(d) < 1;
    sa = 1./(1 + exp(-a(k)));           % sigmoid(a) = softplus'(a)
    dk = d(k);
    r  = sa;                            % the limit at d = 0
    nz = dk ~= 0;
    r(nz) = log1p(sa(nz).*expm1(dk(nz)))./dk(nz);
    s(k)  = r;
end
