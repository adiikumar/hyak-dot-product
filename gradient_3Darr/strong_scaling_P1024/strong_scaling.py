import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator

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
   
fig, ax = plt.subplots()    
plt.loglog(num_nodes, np.ones(len(num_nodes)) * times_CPU_serial, label="CPU_serial")
plt.loglog(num_nodes, times_CPU_MPI, label="CPU_MPI", marker="o")
ax.set_title(
    "Strong scaling of gradient of 3D arrays with grid 1024$^3$;\n"
    "MPI: num_cpus per node = 32;\n"
    "GPUs (using OpenMP): num_gpus per node = 1, 2"
)
plt.xlabel("Number of nodes")
plt.ylabel("Wall time (seconds)")

ax.set_xlim(num_nodes[0] / 1.05, num_nodes[-1] * 1.05)
ax.set_xticks(num_nodes)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())

ax.set_ylim(np.min(times_CPU_MPI) / 1.25, times_CPU_serial * 1.25)
ax.set_yticks(np.concatenate((np.array([times_CPU_serial]), times_CPU_MPI)))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(NullLocator())

plt.tight_layout()

plt.legend()
plt.grid()
plt.savefig("strong_scaling_P1024.svg", dpi=300)