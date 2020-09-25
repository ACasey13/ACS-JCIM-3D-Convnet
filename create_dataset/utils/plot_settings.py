import matplotlib.pyplot as plt

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (6.4, 4.8)
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = ['sans-serif']
#plt.rcParams['font.sans-serif'] = ['HelveticaNeue']
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"


def plt_format(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis='both', labelsize=11)

    xt = ax.get_xticks()
    yt = ax.get_yticks()


    for y in yt[1:-1]:
        ax.plot([xt[0],xt[-1]], [y] * 2, "--", lw=0.5, color="black", alpha=0.3)

    for x in xt[1:-1]:
        ax.plot([x]*2, [yt[0],yt[-1]], "--", lw=0.5, color="black", alpha=0.3)


    ax.tick_params(axis="both", which="both", bottom=False, top=False,
                labelbottom=True, left=False, right=False, labelleft=True)

