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
plt.loglog(num_nodes, np.ones(len(num_nodes)) * times_CPU_serial, label="CPU_serial")
plt.loglog(num_nodes, times_CPU_MPI, label="CPU_MPI", marker="o")
plt.loglog(num_nodes, times_CPU_OMP_T1, label="CPU_OMP_T1", marker="o")
plt.loglog(num_nodes, times_CPU_OMP_T2, label="CPU_OMP_T2", marker="o")
plt.loglog(num_nodes, times_CPU_OMP_T4, label="CPU_OMP_T4", marker="x")
plt.loglog(num_nodes, times_GPU_G1, label="GPU_G1", marker="o")
plt.loglog(num_nodes, times_GPU_G2, label="GPU_G2", marker="o")
plt.loglog(num_nodes, times_GPU_G4, label="GPU_G4", marker="o")
ax.set_title(
    "Strong scaling of gradient of 3D arrays with grid 1024$^3$;\n"
    "MPI: num_cpus per node = 32;\n"
    "GPUs (using OpenMP): num_gpus per node = 1, 2, 4"
)
plt.xlabel("Number of nodes")
plt.ylabel("Wall time (seconds)")

ax.set_xlim(num_nodes[0] / 1.05, num_nodes[-1] * 1.05)
ax.set_xticks(num_nodes)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())

ax.set_ylim(np.min(times_CPU_OMP_T1) / 1.25, times_CPU_serial * 1.25)
ax.set_yticks(np.concatenate((np.array([times_CPU_serial]), times_CPU_OMP_T1, times_CPU_OMP_T2, times_CPU_OMP_T4)))
#ax.set_yticks(np.concatenate((np.array([times_CPU_serial]), times_CPU_OMP_T1, times_CPU_OMP_T2, times_CPU_OMP_T4, times_GPU_G1, times_GPU_G2, times_GPU_G4)))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(NullLocator())

plt.tight_layout()

plt.legend()
plt.grid()
plt.savefig("strong_scaling_P1024.svg", dpi=300)