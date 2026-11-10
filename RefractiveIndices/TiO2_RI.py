import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the file path
file_path = "C:/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/Lab/Ellipsometry/TiO2_mat__files/"

# Define the file names
file_names = [
    "TiO2 (Cauchy).txt",
    "TiO2 (CodyLor).txt",
    "TiO2 (TaucLor).txt",
    "TiO2 1.txt",
    "TiO2 2.txt",
    "TiO2 3.txt",
    "TiO2 extraordinary.txt",
    "TiO2 ordinary.txt",
    "TiO2.txt"
]

# Read the files into DataFrames, skipping the first two rows and without a header
dfs = []
for file_name in file_names:
    df = pd.read_csv(file_path + file_name, sep="\t", header=None, skiprows=2)
    dfs.append(df)

# Convert the DataFrame columns to NumPy arrays
TiO2_Cauchy_nm = np.array(dfs[0][0])
TiO2_Cauchy_n = np.array(dfs[0][1])
TiO2_Cauchy_k = np.array(dfs[0][2])

TiO2_CodyLor_nm = np.array(dfs[1][0])
TiO2_CodyLor_n = np.array(dfs[1][1])
TiO2_CodyLor_k = np.array(dfs[1][2])

TiO2_TaucLor_nm = np.array(dfs[2][0])
TiO2_TaucLor_n = np.array(dfs[2][1])
TiO2_TaucLor_k = np.array(dfs[2][2])

TiO2_1_nm = np.array(dfs[3][0])
TiO2_1_n = np.array(dfs[3][1])
TiO2_1_k = np.array(dfs[3][2])

TiO2_2_nm = np.array(dfs[4][0])
TiO2_2_n = np.array(dfs[4][1])
TiO2_2_k = np.array(dfs[4][2])

TiO2_3_nm = np.array(dfs[5][0])
TiO2_3_n = np.array(dfs[5][1])
TiO2_3_k = np.array(dfs[5][2])

TiO2_extraordinary_nm = np.array(dfs[6][0])
TiO2_extraordinary_n = np.array(dfs[6][1])
TiO2_extraordinary_k = np.array(dfs[6][2])

TiO2_ordinary_nm = np.array(dfs[7][0])
TiO2_ordinary_n = np.array(dfs[7][1])
TiO2_ordinary_k = np.array(dfs[7][2])

TiO2_nm = np.array(dfs[8][0])
TiO2_n = np.array(dfs[8][1])
TiO2_k = np.array(dfs[8][2])

# Reference anatase TiO2
file_path2= 'C:/Users/SchwarzN/OneDrive - Université de Fribourg/Institution/P1_BraggSpericalPigments/Simulation/RefractiveIndices/'
TiO2 = np.genfromtxt(file_path2 + 'TiO2_anatase.txt', delimiter='\t', skip_header=1)
wl_TiO2_nm = TiO2[:, 0]*1e3
n_TiO2 = TiO2[:, 1]

# Plotting
fig, ax1 = plt.subplots()

# Plot _n on the left y-axis
ax1.plot(wl_TiO2_nm, n_TiO2, label='anatase', linewidth=2, color='black', linestyle="--")
ax1.plot(TiO2_Cauchy_nm, TiO2_Cauchy_n, label='Cauchy')
ax1.plot(TiO2_CodyLor_nm, TiO2_CodyLor_n, label='CodyLor')
ax1.plot(TiO2_TaucLor_nm, TiO2_TaucLor_n, label='TaucLor')
ax1.plot(TiO2_1_nm, TiO2_1_n, label='1')
ax1.plot(TiO2_2_nm, TiO2_2_n, label='2')
ax1.plot(TiO2_3_nm, TiO2_3_n, label='3')
ax1.plot(TiO2_extraordinary_nm, TiO2_extraordinary_n, label='extraordinary')
# ax1.plot(TiO2_ordinary_nm, TiO2_ordinary_n, label='ordinary')
ax1.plot(TiO2_nm, TiO2_n, label='TiO2')

ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Refractive Index (n)')
# ax1.legend(loc='upper left')
ax1.legend(loc='upper left', bbox_to_anchor=(-0.5, 1), borderaxespad=0)

# Create a second y-axis for _k
ax2 = ax1.twinx()
ax2.plot(TiO2_Cauchy_nm, TiO2_Cauchy_k, linestyle='--', label='Cauchy')
ax2.plot(TiO2_CodyLor_nm, TiO2_CodyLor_k, linestyle='--', label='CodyLor')
ax2.plot(TiO2_TaucLor_nm, TiO2_TaucLor_k, linestyle='--', label='TaucLor')
ax2.plot(TiO2_1_nm, TiO2_1_k, linestyle='--', label='1')
ax2.plot(TiO2_2_nm, TiO2_2_k, linestyle='--', label='2')
ax2.plot(TiO2_3_nm, TiO2_3_k, linestyle='--', label='3')
ax2.plot(TiO2_extraordinary_nm, TiO2_extraordinary_k, linestyle='--', label='extraordinary')
ax2.plot(TiO2_ordinary_nm, TiO2_ordinary_k, linestyle='--', label='ordinary')
ax2.plot(TiO2_nm, TiO2_k, linestyle='--', label='TiO2')

ax2.set_ylabel('Extinction Coefficient (k)')
# ax2.legend(loc='upper right')
ax2.legend(loc='upper right', bbox_to_anchor=(1.5, 1), borderaxespad=0)

plt.title('Refractive Indices of TiO2')
plt.show()
fig.savefig('RefractiveIndex_TiO2.png', dpi=300, bbox_inches='tight')
