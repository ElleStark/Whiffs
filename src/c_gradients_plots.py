# Script to generate spatial concentration gradients and compare to ftle field
# Elle Stark October 2024

import cmasher as cmr
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

filename = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'

time = 287
xlims = slice(None, None)
ylims = slice(None, None)


with h5py.File(filename, 'r') as f:
    ftle = f.get('FTLE_back_1_25s_finegrid')[time, ylims, xlims]
    # strain = f.get('maxPstrain')[time+30, xlims, ylims]
    # print(snapshot.shape)

file2 = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'

with h5py.File(file2, 'r') as f2:
    odor = f2.get('Odor Data/c')[time+62, xlims, ylims].T
#     print(odor.shape)
    # u = f2.get('Flow Data/u')[time+62, xlims, ylims].T
    # v = f2.get('Flow Data/v')[time+62, xlims, ylims].T
    x_grid = f2.get('Model Metadata/xGrid')[xlims, ylims].T
    y_grid = f2.get('Model Metadata/yGrid')[xlims, ylims].T

odor = np.flipud(odor)

# Compute velocity magnitude
# u_tot = np.sqrt(u**2 + v**2)

# Compute spatial concentration gradient as max of dC/dx or dC/dy
grad_y, grad_x = np.gradient(odor)
# grad_y = np.flipud(abs(grad_y))
# grad_x = np.flipud(abs(grad_x))
# # max_gradient = np.maximum(grad_y, grad_x)
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


# # plot FTLE field
fig, ax = plt.subplots()
colormap = plt.cm.Greys
colormap.set_over('black')
vmin = 0
vmax = 5
# plt.pcolormesh(FTLE_x[:, :], FTLE_y[:, :], ftle[:-1, :-1], norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=colormap, alpha=0.7, linewidths=0)
plt.pcolormesh(FTLE_x[:, :], FTLE_y[:, :], ftle[:-1, :-1], vmin=vmin, vmax=vmax, cmap=colormap)
# plt.colorbar()

# Plot concentration gradient results
# odor[np.where(odor<1*0.0001)] = 1*0.0001
vmin = 1*0.0001
vmax = 0.5
colormap = plt.cm.Reds
colormap.set_under('white')
# plt.pcolormesh(x_grid[:, :], y_grid[:, :], odor[:, :], cmap=plt.cm.Reds, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
plt.pcolormesh(x_grid[:, :], y_grid[:, :], tot_gradient[:, :], cmap=plt.cm.Reds, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.4)
# plt.colorbar()



# Plot velocity magnitude & arrows
# cmap = cmr.waterlily_r
# plt.pcolormesh(x_grid[:, :], y_grid[:, :], u_tot[:, :], cmap=cmap, vmin=-0.25, vmax=0.25)
# plt.colorbar()
# n = 150
# ny = 100
# plt.quiver(x_grid[::ny, ::n], y_grid[::ny, ::n], u[::ny, ::n], v[::ny, ::n], headwidth=6, headlength=7)


# Plot max principal strain
# vmin = 0.5
# vmax = 10
# # plt.pcolormesh(x_grid[:, :], y_grid[:, :], strain[:, :], cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
# plt.pcolormesh(x_grid[:, :], y_grid[:, :], strain[:, :], cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
# plt.colorbar()

ax.set_aspect('equal', adjustable='box')

# plt.savefig('C:/Users/elles/Documents/CU_Boulder/Fluids_Research/Whiffs/ignore/plots/ftle_odor_extended_t287.png', dpi=1000)
plt.show()

