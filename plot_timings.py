import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("timings.csv")

# Convert data_size to numeric in case it's in scientific notation
df["data_size"] = pd.to_numeric(df["data_size"])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each method
for method in df["method"].unique():
    subset = df[df["method"] == method]
    plt.plot(subset["data_size"], subset["time_sec"], marker="o", label=method)


# Set logarithmic scale for x-axis
plt.xscale("log")

# Add labels and title
plt.xlabel("Data Size (log scale)")
plt.ylabel("Time (seconds)")

plt.title("MPI Benchmark: Time vs. Data Size")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("mpi_timings_plot.png")

# Show the plot
plt.show()

