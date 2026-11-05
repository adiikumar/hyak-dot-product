import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator, LogLocator, NullFormatter

def read_application_seconds(path):
    text = Path(path).read_text()
    match = re.search(
        r"^\s*Application_seconds\s*=\s*([-+\d.EeDd]+)",
        text,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Application_seconds not found in {path}")
    # Fortran may write exponents with D instead of E
    s = match.group(1).replace("D", "E").replace("d", "e")
    return np.float64(s)

# READ CPU_SERIAL
times_CPU_serial = read_application_seconds("P1024_CPU_serial.out")

'''ADD MORE DATA POINTS IF NEEDED'''
num_nodes = np.array([1, 2, 4, 8])

# READ CPU_MPI
times_CPU_MPI = np.zeros(len(num_nodes))
CperN = 32.
for i, N in enumerate(num_nodes):
    times_CPU_MPI[i] = read_application_seconds("P1024_CPU_MPI_N" + str(N) + "-C" + str(int(N * CperN)) + ".out")

times_CPU_OMP_T1 = np.zeros(len(num_nodes))
for i, N in enumerate(num_nodes):
    times_CPU_OMP_T1[i] = read_application_seconds("P1024_CPU_OMP_N" + str(N) + "-C" + str(int(N * CperN)) + "-T1.out")

times_CPU_OMP_T2 = np.zeros(len(num_nodes))
for i, N in enumerate(num_nodes):
    times_CPU_OMP_T2[i] = read_application_seconds("P1024_CPU_OMP_N" + str(N) + "-C" + str(int(N * CperN)) + "-T2.out")

times_CPU_OMP_T4 = np.zeros(len(num_nodes))
for i, N in enumerate(num_nodes):
    times_CPU_OMP_T4[i] = read_application_seconds("P1024_CPU_OMP_N" + str(N) + "-C" + str(int(N * CperN)) + "-T4.out")

times_GPU_G1 = np.full(len(num_nodes), np.nan)
times_GPU_G2 = np.full(len(num_nodes), np.nan)
times_GPU_G4 = np.full(len(num_nodes), np.nan)
for i, N in enumerate(num_nodes):
    for G, times in ((1, times_GPU_G1), (2, times_GPU_G2), (4, times_GPU_G4)):
        path = Path("P1024_GPU_N" + str(N) + "-G" + str(G) + ".out")
        if path.exists():
            if "Verification: PASS" not in path.read_text():
                raise ValueError(f"Verification failed: {path}")
            times[i] = read_application_seconds(path)

fig, ax = plt.subplots()

# perfect strong scaling: slope -1 through the 1-node MPI time
ideal = times_CPU_MPI[0] * num_nodes[0] / num_nodes
ax.loglog(num_nodes, ideal, "k--", linewidth=1, label="Ideal strong scaling")

# perfect weak scaling: flat line from P=1 to P=8 at the 1-node MPI time
ax.loglog([1, 8], [times_CPU_MPI[0]] * 2, "k:", linewidth=1, label="Ideal weak scaling")

# (label, data, color, marker)
series = [
    ("MPI (#CPUs/#Nodes=32)",                  times_CPU_MPI,    "C1", "+"),
    #
    ("MPI + OMP(#Threads/#CPUs=1)",    times_CPU_OMP_T1, "C2", "o"),
    ("MPI + OMP(#Threads/#CPUs=2)",   times_CPU_OMP_T2, "C3", "o"),
    ("MPI + OMP(#Threads/#CPUs=4)",   times_CPU_OMP_T4, "C4", "o"),
    #
    ("MPI + OMP(#GPUs/#Nodes=1)",  times_GPU_G1,     "C5", "x"),
    ("MPI + OMP(#GPUs/#Nodes=2)", times_GPU_G2,     "C6", "x"),
    ("MPI + OMP(#GPUs/#Nodes=4)", times_GPU_G4,     "C7", "x"),
]
for label, t, color, marker in series:
    ax.loglog(num_nodes, t, label=label, color=color, marker=marker)

ax.set_title(
    "Scaling of the gradient of 3D array with problem size 1024$^3$"
)
ax.set_xlabel("#Nodes")
ax.set_ylabel("Elapsed wall time (seconds)")

# x: plain 1, 2, 4, 8
ax.set_xlim(num_nodes[0] / 1.05, num_nodes[-1] * 1.05)
ax.set_xticks(num_nodes)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())

# y: crop to data + ideal line, plain decimal labels, unlabeled minor ticks
y = np.concatenate([t for _, t, _, _ in series] + [ideal])
ax.set_ylim(np.nanmin(y) / 1.15, np.nanmax(y) * 1.15)
ax.set_yticks([0.3, 0.5, 1, 2])              # edit if your range changes
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
ax.yaxis.set_minor_formatter(NullFormatter())

# paper style: full box, inward ticks on all four sides
ax.tick_params(which="both", direction="in", top=True, right=True)

ax.grid(which="major", alpha=0.5)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
          fontsize=8, frameon=False)
plt.savefig("strong_scaling_P1024.svg", bbox_inches="tight")