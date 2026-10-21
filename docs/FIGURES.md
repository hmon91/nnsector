# README figures

These original figures explain nnsector using its public API. Numerical
plots use the bundled `nnsector.demo_network()`; the overview and sign-region
figure also contain explanatory schematics. No auto_LiRPA artwork is reused.

## Assets

| File stem in `docs/assets/` | Explanation | Formats |
|---|---|---|
| `overview` | Network → computed sector → feedback-system analysis | PNG, SVG |
| `local-vs-global` | Global and local sectors on identical axes | PNG, SVG |
| `operating-point` | Sectors anchored at 0 and 0.5 | PNG, SVG |
| `domain-and-tightness` | Radius sweep and comparison of nnsector's two interval modes | PNG, SVG |
| `multi-input` | Four sign pieces for a two-input box | PNG, SVG |
| `local-sector` | Recomputed local sector as the input-box radius changes | GIF |

The README embeds PNGs for compatibility. Use the SVG versions for scaling
or reuse in slides. Fonts are embedded as vector paths for portable rendering;
edit labels and styling in the Python source to regenerate them.

## Reproduce

From the nnsector repository root, with this `docs` directory in place:

```bash
python -m pip install -e ".[examples]"
python -m pip install pillow
python docs/make_readme_figures.py
```

The script writes five PNG/SVG pairs, a 57-frame GIF, and
`docs/figure-data.json`. It uses only NumPy, Matplotlib, Pillow, and nnsector.
Run it after updating the library to refresh the numerical figures. Update
the README's printed numbers as well if the computed results change.

## Numerical interpretation

- All sectors use the actual network output at the anchor. In this snapshot,
  `pi(0) = 0.0007229727295428052`, so the demo is not assumed to vanish at zero.
- Scalar sectors are drawn between the minimum and maximum of their two
  boundary lines. This handles the reversal of line ordering to the left
  of the anchor.
- The global sector is valid for every input pair. Its figure displays the
  slice with anchor 0 and inputs in `[-1, 1]`.
- Local sectors hold relative to the selected fixed anchor, within the
  selected box. They are not presented as all-pairs local incremental bounds.
- The dense-grid secant ranges are sampled references, not proven extrema.
  The interval-refinement comparison uses two settings of nnsector. It is
  not a benchmark against auto_LiRPA or another package.
- The 49.3% figure is the reduction in scalar slope-interval width on this
  demo: `100 * (1 - width_default / width_ibp)`.
- The two-input figure illustrates the box `[0, 2] × [0, 1]` with anchor
  `(1, 0.5)`, matching the README example. Its colors identify input sign
  pieces, not sampled output ranges.

## Source and validation

Prepared against [hmon91/nnsector, commit
1b722407aa35c729774c4a943e4c732817ecc633](https://github.com/hmon91/nnsector/tree/1b722407aa35c729774c4a943e4c732817ecc633)
on 2026-10-16. The presentation takes inspiration from the visual-first
organization of [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA).

- Executed the four NumPy examples in the rewritten README and checked
  their local bounds with `nnsector.check`.
- Executed the repository's existing Python examples 1, 2, and 3.
- Checked the default, IBP-interval, shifted-anchor, and global demo bounds
  on 121,003 sample points each using the package's numerical tolerance.
- Checked plotted curves against the displayed sectors and sampled secants
  against the computed intervals throughout the 100-radius sweep.
- Inspected all five static figures and representative animation frames.
- PyTorch, MATLAB, GNU Octave, and pytest were unavailable in the authoring
  environment. These checks were run afterwards, on 2026-10-16 at commit
  [7fc2a78](https://github.com/hmon91/nnsector/tree/7fc2a786cfd11a6a3c1b766c02d78371bbf47667),
  with Python 3.13.16, NumPy 2.5.4, PyTorch 2.14.1 (CPU), pytest 9.1.1, and
  GNU Octave 8.4.0. All of them passed:
  - the PyTorch example in the README and `examples/example4_pytorch.py`
    ran without errors;
  - the MATLAB example in the README ran in GNU Octave. It prints the
    sector `[-2.7863, -1.2002]`, and `nnsector.check` passes;
  - the MATLAB test suite (`matlab/tests/run_tests.m`) passed all 29 tests
    in GNU Octave;
  - the Python test suite passed all 47 tests with pytest, including the 7
    PyTorch tests, with none skipped.

Exact plot data and numerical-check results are in
[`figure-data.json`](figure-data.json). Example execution results,
including those of the later PyTorch, GNU Octave, and pytest checks, are in
[`validation.json`](validation.json).
