import glob
import re

# Find the newest strong-scaling result file
files = glob.glob("results/phase3/pcg_strong_*.out")

if not files:
    print("No strong-scaling result file found")
    exit(1)

file_name = max(files, key=lambda f: int(re.search(r"_(\d+)\.out$", f).group(1)))

data = {}
cur_p = 0

# Read process counts and runtimes
with open(file_name, "r") as file:
    for line in file:
        if line.startswith("Processes:"):
            cur_p = int(line.split(":")[1])

        if line.startswith("Runtime:"):
            time = float(line.split(":")[1])

            if cur_p not in data:
                data[cur_p] = []

            data[cur_p].append(time)

print("File:", file_name)
print()
print("Processes  Runs  Avg time(s)  Speedup  Efficiency(%)")

if 1 not in data:
    print("The 1-process result is incomplete")
    exit(1)

base_time = sum(data[1]) / len(data[1])

for p in sorted(data):
    avg_time = sum(data[p]) / len(data[p])
    speedup = base_time / avg_time
    efficiency = speedup / p * 100.0

    print(
        f"{p:<9}  "
        f"{len(data[p]):<4}  "
        f"{avg_time:<11.6f}  "
        f"{speedup:<7.3f}  "
        f"{efficiency:.2f}"
    )
