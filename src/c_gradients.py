# Script to generate spatial concentration gradients and compare to ftle field
# Elle Stark October 2024

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

filename = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'

time = 2500
xlims = slice(None, None)
ylims = slice(None, None)


with h5py.File(filename, 'r') as f:
    ftle = f.get('FTLE_back_1_25s_finegrid')[time, ylims, xlims]
    # strain = f.get('maxPstrain')[time+30, xlims, ylims]
    # print(snapshot.shape)

file2 = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'

with h5py.File(file2, 'r') as f2:
    odor = f2.get('Odor Data/c')[time+62, xlims, ylims].T
    print(odor.shape)
    x_grid = f2.get('Model Metadata/xGrid')[xlims, ylims].T
    y_grid = f2.get('Model Metadata/yGrid')[xlims, ylims].T

# Compute spatial concentration gradient as max of dC/dx or dC/dy
grad_y, grad_x = np.gradient(odor)
grad_y = np.flipud(abs(grad_y))
grad_x = np.flipud(abs(grad_x))
# max_gradient = np.maximum(grad_y, grad_x)
tot_gradient = np.sqrt(grad_x**2 + grad_y**2)

# Compute spatial FTLE gradient as max of d(FTLE)/dx or d(FTLE)/dy
# ftle_grad_y, ftle_grad_x = np.gradient(ftle)
# ftle_grad_x = abs(ftle_grad_x)
# ftle_grad_y = abs(ftle_grad_y)
# ftle_max_gradient = np.maximum(ftle_grad_x, ftle_grad_y)

# Create half-size mesh grid for FTLE field plotting
FTLE_x = np.linspace(x_grid[0, 0], x_grid[0, -1], len(ftle[0, :])-1)
FTLE_y = np.linspace(y_grid[0, 0], y_grid[-1, 0], len(ftle[:, 0])-1)
FTLE_x, FTLE_y = np.meshgrid(FTLE_x, FTLE_y)

# plot FTLE field
fig, ax = plt.subplots()
colormap = plt.cm.Greys
vmin = 0
vmax = 1
# plt.pcolormesh(FTLE_x[:, 1500:], FTLE_y[:, 1500:], ftle[:-1, 1500:-1], norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=colormap, alpha=0.7, linewidths=0)
plt.pcolormesh(FTLE_x[:, 1500:], FTLE_y[:, 1500:], ftle[:-1, 1500:-1], vmin=vmin, vmax=vmax, cmap=colormap, alpha=0.7, linewidths=0)
plt.colorbar()

# Plot concentration gradient results
vmin = 0.0001
vmax = 0.01
plt.pcolormesh(x_grid[:, 750:], y_grid[:, 750:], tot_gradient[:, 750:], cmap=plt.cm.Reds, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
plt.colorbar()

# Plot max principal strain
# vmin = 1
# vmax = 5
# plt.pcolormesh(x_grid[:, 750:], y_grid[:, 750:], strain[:, 750:], cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
# plt.colorbar()

ax.set_aspect('equal', adjustable='box')

plt.show()

