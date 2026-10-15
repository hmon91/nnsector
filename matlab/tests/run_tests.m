function run_tests()
%RUN_TESTS  Tests for the nnsector library.
%   Run with nnsector/matlab/tests as the current folder (or on the path):
%       run_tests
%   Prints PASS or FAIL for each test and ends with an error if any failed.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(root);
    net = nnsector.demo_network();      % 1 input, 8 and 8 tanh neurons, biases
    W   = net.W;

    st = rng;  rng(0);
    restore = onCleanup(@() rng(st));   %#ok<NASGU>
    nfail = 0;

    % ---- the demo network ---------------------------------------------------
    sb = nnsector.local_bound(net, -1, 1);
    nfail = nfail + report('demo network: local bound about y* = 0 on [-1, 1] is [-2.7863, -1.2002]', ...
        numel(sb.pieces) == 2 && near(sb.gamma1, -2.78625, 1e-5) && near(sb.gamma2, -1.20015, 1e-5));
    nfail = nfail + report('demo network: no neuron in Case 3', ...
        isequal([sb.pieces.case3], [0 0 0 0]));

    y = linspace(-1, 1, 200001);  y = y(y ~= 0);
    r = (nnsector.forward(net, y) - sb.pistar) ./ y;
    nfail = nfail + report('demo network: the bound contains the exact slopes [-2.5200, -1.4630]', ...
        sb.gamma1 <= min(r) && max(r) <= sb.gamma2 && ...
        near(min(r), -2.51995, 1e-5) && near(max(r), -1.46305, 1e-5));

    sbi = nnsector.local_bound(net, -1, 1, 'Intervals', 'ibp');
    nfail = nfail + report('demo network with plain IBP intervals is [-2.8782, 0.2514]', ...
        near(sbi.gamma1, -2.87818, 1e-5) && near(sbi.gamma2, 0.25138, 1e-5));

    gb = nnsector.global_bound(net);
    nfail = nfail + report('demo network: global bound is [-4.6588, 4.6588]', ...
        near(gb.Gamma2, 4.65881, 1e-5) && gb.Gamma1 == -gb.Gamma2);

    % ---- y* inside the box, and at its upper end ---------------------------
    sb1 = nnsector.local_bound(net, -1, 1, 'ystar', 0.5);
    yy  = linspace(-1, 1, 200001);
    yy  = yy(abs(yy - 0.5) > 1e-6);
    r1  = (nnsector.forward(net, yy) - sb1.pistar) ./ (yy - 0.5);
    nfail = nfail + report('y* = 0.5: two pieces, all secant slopes in [gamma1, gamma2]', ...
        numel(sb1.pieces) == 2 && sb1.gamma1 <= min(r1) && max(r1) <= sb1.gamma2 && ...
        near(sb1.gamma1, -2.37991, 1e-5) && near(sb1.gamma2, -0.75116, 1e-5) && ...
        passes(net, sb1));

    sb2 = nnsector.local_bound(net, -1, 1, 'ystar', 1);
    nfail = nfail + report('y* = 1 (upper end): one piece, with sign -1', ...
        numel(sb2.pieces) == 1 && sb2.pieces.signs == -1 && passes(net, sb2));

    % ---- random networks with biases ---------------------------------------
    ok = true;
    for t = 1:150
        p = randi(3);  m = randi(2);
        widths = [p, randi([2 10], 1, randi(3)), m];
        Wr = cell(1, numel(widths)-1);  br = Wr;
        for k = 1:numel(Wr)
            Wr{k} = randn(widths(k+1), widths(k));
            br{k} = randn(widths(k+1), 1);
        end
        netr = nnsector.network(Wr, br);
        lo = -2 + 2*rand(p,1);
        hi = lo + 0.1 + 2*rand(p,1);
        ys = lo + (hi - lo).*rand(p,1);
        if mod(t,5) == 0, ys = lo; end          % y* at the lower corner
        for mode = {'intersect', 'ibp'}
            sbr = nnsector.local_bound(netr, lo, hi, 'ystar', ys, 'Intervals', mode{1});
            ok  = ok && passes(netr, sbr, 4000);
        end
        gbr = nnsector.global_bound(netr);
        ok  = ok && passes(netr, gbr, 4000, 'Box', [lo hi], 'ystar', ys);
    end
    nfail = nfail + report('150 random networks with biases: every bound holds on samples', ok);

    % ---- other activations --------------------------------------------------
    acts = {{'tanh'}, {'sigmoid'}, {'relu'}, {'leakyrelu', 0.01}, ...
            {'leakyrelu', 0.3}, {'leakyrelu', 2}, {'softplus'}};

    % secant slope ranges against brute force, with end points at a and kinks
    ok = true;  gap = 0;
    for ia = 1:numel(acts)
        A = nnsector.activation(acts{ia}{:});
        for t = 1:400
            a = 2*randn;  if rand < 0.15, a = 0; end
            x = sort(a + 3*randn(2,1));
            if rand < 0.25, x(randi(2)) = a;  x = sort(x); end
            [s_inf, s_sup] = A.secant_range(a, x(1), x(2));
            xs = linspace(x(1), x(2), 2001);
            xs = xs(abs(xs - a) > 1e-5*(1 + abs(a)));      % no rounding blow-up
            if isempty(xs), continue; end
            s   = (A.f(xs) - A.f(a)) ./ (xs - a);
            ok  = ok && s_inf <= min(s) + 1e-9 && max(s) <= s_sup + 1e-9;
            gap = max([gap, min(s) - s_inf, s_sup - max(s)]);
        end
    end
    nfail = nfail + report('secant slope ranges contain all sampled slopes, every activation', ok);
    nfail = nfail + report('secant slope ranges are tight (gap to the samples below 1e-3)', gap < 1e-3);

    % one neuron, y* = 0 inside the box: the bound is the exact sector
    sp = @(x) log(1 + exp(x));
    one = { 'relu',      [],   [-1 2], [0, 1]
            'leakyrelu', 0.1,  [-1 2], [0.1, 1]
            'sigmoid',   [],   [-1 1], [1/(1+exp(-1)) - 0.5, 0.25]
            'softplus',  [],   [-1 1], [sp(0) - sp(-1), sp(1) - sp(0)] };
    ok = true;
    for k = 1:size(one,1)
        net1 = nnsector.network({1, 1}, [], 'Activation', one{k,1}, 'Alpha', one{k,2});
        sb1n = nnsector.local_bound(net1, one{k,3}(1), one{k,3}(2));
        ok   = ok && near([sb1n.gamma1, sb1n.gamma2], one{k,4}, 1e-12);
    end
    nfail = nfail + report('one neuron of each new activation: exact sector', ok);

    % 0 <= relu(y1 - y2) <= y1 on [0, 1]^2: in Case 3 the slope of ReLU
    % left of nu* = 0 is 0, and max(U, 0) drops y2, which can only lower it
    netc = nnsector.network({[1 -1], 1}, [], 'Activation', 'relu');
    sbc  = nnsector.local_bound(netc, [0; 0], [1; 1]);
    nfail = nfail + report('ReLU in Case 3: 0 <= relu(y1 - y2) <= y1', ...
        isequal(sbc.gamma1, [0 0]) && isequal(sbc.gamma2, [1 0]));

    % random networks with biases for every activation
    for ia = 1:numel(acts)
        ok = true;
        for t = 1:40
            p = randi(3);  m = randi(2);
            widths = [p, randi([2 10], 1, randi(3)), m];
            Wr = cell(1, numel(widths)-1);  br = Wr;
            for k = 1:numel(Wr)
                Wr{k} = randn(widths(k+1), widths(k));
                br{k} = randn(widths(k+1), 1);
            end
            alpha = [];  if numel(acts{ia}) > 1, alpha = acts{ia}{2}; end
            netr = nnsector.network(Wr, br, 'Activation', acts{ia}{1}, 'Alpha', alpha);
            lo = -2 + 2*rand(p,1);
            hi = lo + 0.1 + 2*rand(p,1);
            ys = lo + (hi - lo).*rand(p,1);
            for mode = {'intersect', 'ibp'}
                sbr = nnsector.local_bound(netr, lo, hi, 'ystar', ys, 'Intervals', mode{1});
                ok  = ok && passes(netr, sbr, 3000);
            end
            gbr = nnsector.global_bound(netr);
            ok  = ok && passes(netr, gbr, 3000, 'Box', [lo hi], 'ystar', ys);
        end
        name = acts{ia}{1};
        if numel(acts{ia}) > 1, name = sprintf('%s (alpha = %g)', name, acts{ia}{2}); end
        nfail = nfail + report(sprintf('%s: 40 random networks with biases, bounds hold', name), ok);
    end

    nets = nnsector.network(W, [], 'Activation', 'sigmoid');
    gbs  = nnsector.global_bound(nets);
    nfail = nfail + report('sigmoid global bound uses c = 1/4', ...
        gbs.c == 0.25 && near(gbs.Gamma2, 0.25^2*gb.Gamma2, 1e-12));

    % ---- special cases ------------------------------------------------------
    Wl   = [1 -2; 3 0.5];
    netl = nnsector.network({Wl}, {[1; -1]});              % no hidden layer
    sbl  = nnsector.local_bound(netl, [-1; 0], [1; 2], 'ystar', [0; 1]);
    okl  = numel(sbl.pieces) == 4;
    for k = 1:numel(sbl.pieces)
        P   = sbl.pieces(k);
        okl = okl && isequal(P.L, P.U) && isequal(P.L, Wl*diag(P.signs));
    end
    nfail = nfail + report('linear network, y* inside: 4 pieces, each bound exact', okl);

    h  = 1e-6;
    dp = (nnsector.forward(net, 1 + h) - nnsector.forward(net, 1 - h)) / (2*h);
    sbd = nnsector.local_bound(net, 1, 1, 'ystar', 1);    % the box is one point
    nfail = nfail + report('one-point box gives the derivative of the network', ...
        near(sbd.gamma1, dp, 1e-6) && near(sbd.gamma2, dp, 1e-6));

    % every neuron of these networks is monotone in y, so no interval
    % straddles nu*, and rounding must not put a neuron in Case 3
    okm = true;
    for t = 1:20
        s1 = sign(randn(6,1));  r3 = sign(randn(4,1));
        Wm = {s1.*abs(randn(6,1)), abs(randn(5,6)).*s1', r3.*abs(randn(4,5)), randn(1,4)};
        bm = {randn(6,1), randn(5,1), randn(4,1), 0};
        for ia = 1:numel(acts)
            alpha = [];  if numel(acts{ia}) > 1, alpha = acts{ia}{2}; end
            netm = nnsector.network(Wm, bm, 'Activation', acts{ia}{1}, 'Alpha', alpha);
            for mode = {'intersect', 'ibp'}
                sbm = nnsector.local_bound(netm, -1, 1.5, 'ystar', 0.3, 'Intervals', mode{1});
                okm = okm && all([sbm.pieces.case3] == 0);
            end
        end
    end
    nfail = nfail + report('monotone networks: no neuron in Case 3', okm);

    % ---- input errors -------------------------------------------------------
    nfail = nfail + report('y* outside the box is an error', ...
        raises(@() nnsector.local_bound(net, 0, 2, 'ystar', 3), 'nnsector:local_bound:ystar'));
    nfail = nfail + report('default y* = 0 outside the box is an error', ...
        raises(@() nnsector.local_bound(net, 1, 2), 'nnsector:local_bound:ystar'));
    nfail = nfail + report('weights that do not fit together are an error', ...
        raises(@() nnsector.network({ones(3,2), ones(1,4)}), 'nnsector:network:size'));
    nfail = nfail + report('an unsupported activation is an error', ...
        raises(@() nnsector.network(W, [], 'Activation', 'swish'), ...
               'nnsector:activation:unsupported'));
    nfail = nfail + report('a negative leaky ReLU slope is an error', ...
        raises(@() nnsector.network(W, [], 'Activation', 'leakyrelu', 'Alpha', -0.1), ...
               'nnsector:activation:alpha'));
    ws = warning('error', 'nnsector:network:alpha');    %#ok<CTPCT> catch the warning
    warned = raises(@() nnsector.network(W, [], 'Activation', 'relu', 'Alpha', 0.2), ...
                    'nnsector:network:alpha');
    warning(ws);
    nfail = nfail + report('Alpha with an activation other than leakyrelu warns', warned);

    if nfail > 0
        error('nnsector:tests', '%d test(s) failed.', nfail);
    end
    fprintf('All tests passed.\n');
end

% -------------------------------------------------------------------------
function f = report(name, ok)
    if ok
        fprintf('PASS  %s\n', name);
    else
        fprintf('FAIL  %s\n', name);
    end
    f = double(~ok);
end

function t = near(a, b, tol)
    if nargin < 3, tol = 5e-5; end
    t = all(abs(a(:) - b(:)) <= tol);
end

function t = passes(net, bound, n, varargin)
    if nargin < 3, n = 20000; end
    r = nnsector.check(net, bound, 'Samples', n, varargin{:});
    t = r.passed;
end

function t = raises(f, id)
    try
        f();
        t = false;
    catch err
        t = strcmp(err.identifier, id);
    end
end
