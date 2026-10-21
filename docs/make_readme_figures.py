"""Reproduce nnsector's README figures using its public API.

Run from a checkout with nnsector installed:
    python -m pip install matplotlib pillow
    python docs/make_readme_figures.py

All numeric plots use nnsector.demo_network(). No fitted or hand-drawn
bounds are substituted for the outputs of nnsector. Sampled secant ranges
are diagnostics, not certified extrema. SVGs are standalone vector assets;
PNGs are the README's widely compatible display assets.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
from PIL import Image

import nnsector as ns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#152B43"
TEAL = "#008C86"
ORANGE = "#CB681C"
BLUE = "#4265B0"
PURPLE = "#8665AA"
INK = "#25364A"
MUTED = "#5C6D7F"
GRID = "#DFE6EC"
PALE = "#F4F7FA"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#B3C0CC",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "path",
    "savefig.facecolor": WHITE,
})

NET = ns.demo_network()
SB = ns.local_bound(NET, -1, 1)
IBP = ns.local_bound(NET, -1, 1, intervals="ibp")
GB = ns.global_bound(NET)
SHIFTED = ns.local_bound(NET, -1, 1, ystar=0.5)


def vals(bound):
    return float(bound.gamma1.item()), float(bound.gamma2.item())


def interval(bound):
    a, b = vals(bound)
    return f"[{a:.4f}, {b:.4f}]"


def canvas(title, subtitle, size=(14, 6.1)):
    fig = plt.figure(figsize=size, facecolor=WHITE)
    fig.text(0.052, 0.935, title, size=23, weight="bold", color=NAVY)
    fig.text(0.052, 0.881, subtitle, size=11.5, color=MUTED)
    return fig


def save(fig, name):
    # Finish rendering each asset in memory before replacing the output file.
    for extension in ("png", "svg"):
        data = io.BytesIO()
        fig.savefig(data, format=extension, dpi=160)
        (OUT / f"{name}.{extension}").write_bytes(data.getvalue())
    plt.close(fig)


def sector_legend(fig):
    fig.legend([Line2D([], [], color=NAVY, lw=2.5),
                Line2D([], [], color=MUTED, lw=1.8, ls="--"),
                Patch(facecolor=TEAL, alpha=.17, edgecolor="none")],
               ["Neural network", "Sector boundaries", "Enclosed output region"],
               loc="center", bbox_to_anchor=(.50, .815), ncol=3,
               frameon=False, fontsize=10.5)


def style(ax, xlabel="Input  $y$", ylabel=r"Network output  $\pi(y)$"):
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="#B3C0CC", lw=0.8)
    ax.axvline(0, color="#B3C0CC", lw=0.8)


def sector(ax, bound, color=TEAL, n=1001, labels=False):
    x = np.linspace(bound.y_lo[0], bound.y_hi[0], n)
    a, b = vals(bound)
    y1 = bound.pistar[0] + a * (x - bound.ystar[0])
    y2 = bound.pistar[0] + b * (x - bound.ystar[0])
    ax.fill_between(x, np.minimum(y1, y2), np.maximum(y1, y2),
                    color=color, alpha=0.17, zorder=2)
    ax.plot(x, y1, color=color, lw=1.8, ls="--")
    ax.plot(x, y2, color=color, lw=1.8, ls="--")
    ax.plot(x, NET(x[:, None])[:, 0], color=NAVY, lw=2.6,
            label="Neural network" if labels else None, zorder=4)
    ax.scatter(bound.ystar, bound.pistar, s=52, facecolor=WHITE,
               edgecolor=NAVY, linewidth=2, zorder=6)
    assert np.all(NET(x[:, None])[:, 0] >= np.minimum(y1, y2) - 1e-8)
    assert np.all(NET(x[:, None])[:, 0] <= np.maximum(y1, y2) + 1e-8)


def arrow(ax, start, end, color=MUTED, lw=1.7, **kw):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            color=color, linewidth=lw, **kw)
    ax.add_patch(patch)


def overview():
    fig = canvas("From a trained network to bounds for control",
                 "Choose an operating region. Compute a sector. Use it in your system analysis.",
                 (15, 6.2))
    base = fig.add_axes([0, 0, 1, 1]); base.axis("off")
    for x, number, heading in [(0.045, "01", "Provide your network"),
                               (0.365, "02", "Compute its sector"),
                               (0.685, "03", "Analyze the feedback loop")]:
        base.add_patch(FancyBboxPatch((x, .19), .285, .60,
                       boxstyle="round,pad=0.008,rounding_size=0.016",
                       facecolor=PALE, edgecolor="none"))
        base.text(x+.015, .738, number, color=TEAL, weight="bold", size=12)
        base.text(x+.049, .738, heading, color=NAVY, weight="bold", size=12)
    nn = fig.add_axes([.068,.315,.24,.325]); nn.axis("off")
    xs = [.08, .36, .64, .92]
    levels = [[.5],np.linspace(.10,.90,8),np.linspace(.10,.90,8),[.5]]
    for i in range(3):
        for y1 in levels[i]:
            for y2 in levels[i+1]:
                nn.plot([xs[i], xs[i+1]], [y1, y2], color="#C4D2DF", lw=.65, zorder=1)
    for i, ys in enumerate(levels):
        for y in ys:
            nn.scatter([xs[i]], [y], s=54, color=TEAL if i in (1,2) else NAVY,
                       edgecolor=WHITE, lw=1, zorder=2)
    nn.set_xlim(0,1); nn.set_ylim(0,1)
    base.text(.188, .262, "Weights + biases + activation",ha="center",size=11,color=INK)
    base.text(.188, .222, "Demo: 1 input · 8 + 8 hidden neurons · 1 output",ha="center",size=8.8,color=MUTED)
    ax=fig.add_axes([.405,.37,.205,.28]); sector(ax,SB)
    style(ax,"Input", "Output")
    ax.set_xticks([-1,0,1]); ax.set_yticks([-2,0,2]); ax.tick_params(labelsize=9)
    ax.set_xlim(-1,1); ax.set_ylim(-3,3)
    ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
    base.text(.508,.262,interval(SB),ha="center",color=TEAL,weight="bold",size=14)
    base.text(.508,.222,"Local secant slopes about y* = 0 on [−1, 1]",ha="center",size=9,color=MUTED)
    loop=fig.add_axes([.703,.325,.25,.30]);loop.set_xlim(0,1);loop.set_ylim(0,1);loop.axis("off")
    for x,label,c in [(.02,"Neural\ncontroller",TEAL),(.60,"Plant",NAVY)]:
        loop.add_patch(FancyBboxPatch((x,.42),.36,.40,boxstyle="round,pad=.01,rounding_size=.02",facecolor=WHITE,edgecolor=c,lw=1.7))
        loop.text(x+.18,.62,label,ha="center",va="center",weight="bold",size=11,color=c)
    arrow(loop,(.39,.62),(.59,.62),NAVY)
    loop.text(.49,.72,"u",ha="center",size=11)
    loop.plot([.78,.78,.20],[.40,.17,.17],color=MUTED,lw=1.5)
    arrow(loop,(.20,.17),(.20,.40))
    loop.text(.49,.04,"measured state / output",ha="center",size=9,color=MUTED)
    base.text(.828,.262,"Combine sector + plant + criterion",ha="center",size=11,color=INK)
    base.text(.828,.222,"For example, stability analysis with nnstab",ha="center",size=9,color=MUTED)
    arrow(base,(.333,.49),(.359,.49),TEAL)
    arrow(base,(.653,.49),(.679,.49),TEAL)
    base.text(.052,.08,"nnsector",weight="bold",color=NAVY,size=20)
    base.text(.19,.085,"Python / NumPy  ·  MATLAB / GNU Octave  ·  Optional PyTorch import",color=MUTED,size=11)
    save(fig,"overview")


def local_global():
    fig=canvas("Local information sharpens the sector",
               "Same demo network, same plotting scale. Both sectors pass through the operating point.")
    sector_legend(fig)
    axes=[fig.add_axes([.09,.29,.37,.43]),fig.add_axes([.58,.29,.37,.43])]
    x=np.linspace(-1,1,1001); ps=SB.pistar[0]; g=GB.Gamma2.item()
    ax=axes[0]
    ax.fill_between(x,ps-g*np.abs(x),ps+g*np.abs(x),color=BLUE,alpha=.14)
    ax.plot(x,ps+g*x,color=BLUE,lw=1.6,ls="--")
    ax.plot(x,ps-g*x,color=BLUE,lw=1.6,ls="--")
    ax.plot(x,NET(x[:,None])[:,0],color=NAVY,lw=2.6)
    ax.scatter([0],[ps],s=50,facecolor=WHITE,edgecolor=NAVY,zorder=5,lw=2)
    ax.set_title("Global bound",loc="left",pad=14)
    axes[1].set_title("Local bound on [−1, 1]",loc="left",pad=14)
    sector(axes[1],SB)
    for ax in axes:
        style(ax);ax.set_xlim(-1,1);ax.set_ylim(-5,5)
        ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([-4,-2,0,2,4])
    fig.text(.275,.13,f"Slope interval  [{-g:.4f}, {g:.4f}]",ha="center",weight="bold",color=BLUE,size=13)
    fig.text(.765,.13,"Slope interval  "+interval(SB),ha="center",weight="bold",color=TEAL,size=13)
    fig.text(.275,.075,"Valid for every pair of inputs; only [−1, 1] is shown.",ha="center",color=MUTED,size=10)
    fig.text(.765,.075,"Valid relative to the fixed anchor y* = 0 on this box.",ha="center",color=MUTED,size=10)
    save(fig,"local-vs-global")


def operating_point():
    fig=canvas("Move the anchor to your operating point",
               "The sector bounds π(y) − π(y*). Biases and a nonzero value of π(y*) are retained.")
    sector_legend(fig)
    for left,b,color in [(.09,SB,TEAL),(.58,SHIFTED,ORANGE)]:
        ax=fig.add_axes([left,.29,.37,.43]);sector(ax,b,color)
        style(ax);ax.set_xlim(-1,1);ax.set_ylim(-3.6,3.6)
        ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([-3,0,3])
        a=b.ystar.item();p=b.pistar.item()
        ax.set_title(f"Anchor y* = {a:g}",loc="left",pad=14)
        ax.annotate("(y*, π(y*))",xy=(a,p),xytext=(.42,2.45),fontsize=11,
                    arrowprops=dict(arrowstyle="->",color=MUTED,lw=1.2),color=MUTED)
        fig.text(left+.185,.13,"Slope interval  "+interval(b),ha="center",color=color,weight="bold",size=13)
    fig.text(.5,.065,"A fixed-anchor sector constrains secants through one point; it is not an all-pairs local slope bound.",
             ha="center",size=11,color=MUTED)
    save(fig,"operating-point")


def secants(radius, anchor=0, n=20001):
    x=np.linspace(-radius,radius,n)
    x=x[np.abs(x-anchor)>1e-7]
    r=(NET(x[:,None])[:,0]-NET(anchor).item())/(x-anchor)
    return float(r.min()),float(r.max())


def domain_tightness():
    radii=np.linspace(.05,2,100)
    computed=np.array([vals(ns.local_bound(NET,-r,r)) for r in radii])
    sampled=np.array([secants(r,n=10001) for r in radii])
    assert np.all(computed[:,0] <= sampled[:,0]+1e-8)
    assert np.all(sampled[:,1] <= computed[:,1]+1e-8)
    fig=canvas("See what the operating region changes",
               "Computed on the bundled tanh network with y* = 0. Sampled ranges are diagnostic references.",(14,6.5))
    ax=fig.add_axes([.085,.27,.39,.45])
    ax.fill_between(radii,computed[:,0],computed[:,1],color=TEAL,alpha=.17)
    ax.plot(radii,computed[:,0],color=TEAL,lw=2)
    ax.plot(radii,computed[:,1],color=TEAL,lw=2)
    ax.plot(radii,sampled[:,0],color=NAVY,lw=1.6,ls=":")
    ax.plot(radii,sampled[:,1],color=NAVY,lw=1.6,ls=":")
    ax.axvline(1,color=MUTED,lw=1,ls="--",alpha=.6)
    ax.set_xlim(.05,2);ax.set_ylim(-3.2,-.3)
    ax.set_xlabel("Box radius r, with y in [−r, r]",labelpad=9)
    ax.set_ylabel("Secant slope",labelpad=9)
    ax.set_title("Smaller boxes give tighter sectors here",loc="left",pad=16)
    ax.grid(color=GRID,lw=.8)
    ax.legend([Line2D([],[],color=TEAL,lw=2),Line2D([],[],color=NAVY,lw=1.6,ls=":")],
              ["Computed sector endpoints","Sampled slope extrema"],loc="upper left",frameon=False,fontsize=10)
    ax2=fig.add_axes([.59,.27,.36,.45])
    rs=secants(1,n=200001)
    specs=[("IBP intervals",vals(IBP),ORANGE,2),
           ("Default: intersection",vals(SB),TEAL,1),
           ("Sampled secants",rs,NAVY,0)]
    for label,(a,b),c,y in specs:
        ax2.plot([a,b],[y,y],color=c,lw=9,solid_capstyle="round",alpha=.9)
        ax2.scatter([a,b],[y,y],s=36,facecolor=WHITE,edgecolor=c,lw=1.6,zorder=4)
        ax2.text(-3.05,y+.30,label,color=c,size=11,weight="bold")
        ax2.text((a+b)/2,y-.29,f"[{a:.3f}, {b:.3f}]",ha="center",size=10,color=MUTED)
    ax2.set_xlim(-3.1,.45);ax2.set_ylim(-.63,2.68);ax2.set_yticks([])
    ax2.set_xticks([-3,-2,-1,0]);ax2.set_xlabel("Secant slope",labelpad=9)
    ax2.spines["left"].set_visible(False)
    ax2.grid(axis="x",color=GRID,lw=.8)
    ax2.set_title("Refine neuron intervals on [−1, 1]",loc="left",pad=16)
    improvement=100*(1-(vals(SB)[1]-vals(SB)[0])/(vals(IBP)[1]-vals(IBP)[0]))
    fig.text(.28,.12,"The same network is used at every radius.",ha="center",size=11,color=MUTED)
    fig.text(.77,.12,f"{improvement:.1f}% narrower slope interval on this example",ha="center",size=12,color=TEAL,weight="bold")
    fig.text(.5,.055,"Both settings use nnsector's sector algorithm; only the neuron-interval construction changes.",ha="center",size=10,color=MUTED)
    save(fig,"domain-and-tightness")
    return radii,computed,sampled,rs,improvement


def multi_input():
    fig=canvas("Several inputs: one bound per sign region",
               "A two-dimensional box split at y* = (1, 0.5). Each piece has its own matrices L and U.",(14,6.4))
    ax=fig.add_axes([.085,.22,.36,.52])
    quadrants=[(1,.5,TEAL,"(+1, +1)"),(0,.5,BLUE,"(−1, +1)"),
               (0,0,PURPLE,"(−1, −1)"),(1,0,ORANGE,"(+1, −1)")]
    for x,y,c,s in quadrants:
        ax.add_patch(Rectangle((x,y),1,.5,facecolor=c,alpha=.12,edgecolor="none"))
        ax.text(x+.5,y+.28,"signs",ha="center",size=11,color=c)
        ax.text(x+.5,y+.17,s,ha="center",size=14,color=c,weight="bold")
    ax.add_patch(Rectangle((0,0),2,1,facecolor="none",edgecolor=NAVY,lw=1.4))
    ax.axvline(1,color=NAVY,lw=1.2,ls="--");ax.axhline(.5,color=NAVY,lw=1.2,ls="--")
    ax.scatter([1],[.5],s=75,color=NAVY,zorder=4)
    ax.text(1.055,.535,"y*",weight="bold",size=14,color=NAVY,
            bbox=dict(facecolor=WHITE,edgecolor="none",pad=2))
    ax.set_xlim(-.03,2.03);ax.set_ylim(-.02,1.02)
    ax.set_xticks([0,1,2]);ax.set_yticks([0,.5,1]);ax.set_xlabel("Input $y_1$");ax.set_ylabel("Input $y_2$")
    ax.spines[["bottom","left"]].set_visible(False)
    right=fig.add_axes([.52,.20,.44,.57]);right.axis("off")
    right.text(0,.94,"Work with nonnegative displacements",size=15,weight="bold",color=NAVY)
    right.text(0,.79,r"$z = |y-y^*| \;\geq\; 0$",size=21,color=TEAL)
    right.text(0,.65,r"$y = y^* + \mathrm{diag}(s)\,z$",size=19,color=NAVY)
    right.text(0,.45,"On each piece:",size=12,color=MUTED)
    right.text(0,.30,r"$L_k z \;\leq\; \pi(y)-\pi(y^*) \;\leq\; U_k z$",size=19,color=NAVY)
    right.text(0,.10,"Read sb.pieces[k].L, .U, .signs and .z_hi",size=12,color=TEAL)
    right.text(0,-.025,"Here sb.gamma1 and sb.gamma2 are None.",size=11,color=MUTED)
    fig.text(.085,.085,r"With $p$ inputs, an interior anchor can create up to $2^p$ pieces. A lower-corner anchor needs one piece.",size=11,color=MUTED)
    save(fig,"multi-input")


def animation():
    # Keep the axes fixed: only the domain and recomputed sector change.
    sequence=np.r_[np.linspace(2,.2,29),np.linspace(.2,2,29)[1:]]
    fig=plt.figure(figsize=(10,5.1),facecolor=WHITE)
    ax=fig.add_axes([.10,.20,.55,.58]); xall=np.linspace(-2,2,1001)
    frames=[]
    for r in sequence:
        fig.clear()
        ax=fig.add_axes([.10,.20,.55,.58])
        fig.text(.055,.91,"Choose a region. Recompute its sector.",size=21,color=NAVY,weight="bold")
        fig.text(.055,.85,"Bundled demo network · fixed anchor y* = 0 · fixed plotting scale",size=10,color=MUTED)
        bound=ns.local_bound(NET,-float(r),float(r))
        ax.plot(xall,NET(xall[:,None])[:,0],color="#BCC7D2",lw=2)
        sector(ax,bound)
        ax.axvline(-r,color=TEAL,lw=1,ls=":");ax.axvline(r,color=TEAL,lw=1,ls=":")
        style(ax)
        ax.set_xlim(-2.1,2.1);ax.set_ylim(-6.5,6.5);ax.set_xticks([-2,-1,0,1,2])
        fig.text(.71,.68,"OPERATING BOX",size=10,color=MUTED,weight="bold")
        fig.text(.71,.59,f"[−{r:.2f}, {r:.2f}]",size=21,color=NAVY,weight="bold")
        fig.text(.71,.45,"SECTOR SLOPES",size=10,color=MUTED,weight="bold")
        a,b=vals(bound)
        fig.text(.71,.37,f"γ₁ = {a:.4f}",size=19,color=TEAL)
        fig.text(.71,.29,f"γ₂ = {b:.4f}",size=19,color=TEAL)
        fig.text(.055,.065,"The shaded sector applies inside the selected box. The gray curve shows the network outside it.",size=10,color=MUTED)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    durations=[100]*len(frames);durations[0]=1000;durations[28]=1300;durations[-1]=600
    frames[0].save(OUT/"local-sector.gif",save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,optimize=True)
    plt.close(fig)


def main():
    overview();local_global();operating_point()
    radii,bounds,samples,rs,improvement=domain_tightness()
    multi_input();animation()
    checks={name:{"passed":bool((c:=ns.check(NET,bound)).passed),
                  "max_violation":float(c.max_violation),"tolerance":float(c.tol),
                  "samples":int(c.samples)}
            for name,bound in [("local",SB),("ibp_intervals",IBP),("shifted_anchor",SHIFTED)]}
    gcheck=ns.check(NET,GB,box=(-20,20))
    checks["global"]={"passed":bool(gcheck.passed),"max_violation":float(gcheck.max_violation),
                      "tolerance":float(gcheck.tol),"samples":int(gcheck.samples)}
    assert all(c["passed"] for c in checks.values())
    data={"network":"nnsector.demo_network()","widths":NET.widths,
          "activation":NET.activation,"pistar_at_zero":NET(0).item(),
          "local_default":vals(SB),"local_ibp_intervals":vals(IBP),
          "global":[GB.Gamma1.item(),GB.Gamma2.item()],"anchor_0_5":vals(SHIFTED),
          "sampled_secant_range":rs,"sampled_range_is_a_proof":False,
          "slope_width_reduction_percent_on_demo":improvement,
          "radius_sweep":{"radius":radii.tolist(),"computed_bounds":bounds.tolist(),
                          "sampled_extrema":samples.tolist()},"numerical_checks":checks}
    (ROOT/"figure-data.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:v for k,v in data.items() if k!="radius_sweep"},indent=2))


if __name__=="__main__":
    main()
