# Script to generate spatial concentration gradients and compare to ftle field
# Elle Stark October 2024

import cmasher as cmr
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

time = 287
xlims = slice(400, 600)
ylims = slice(500, 700)
xlims_ftle = slice(800, 1200)
ylims_ftle = slice(1000, 1400)

filename = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
with h5py.File(filename, 'r') as f:
    ftle = f.get('FTLE_back_1_25s_finegrid')[time, ylims_ftle, xlims_ftle]
    # strain = f.get('maxPstrain')[time+30, xlims, ylims]
    # print(ftle.shape)

file2 = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
with h5py.File(file2, 'r') as f2:
    odor = f2.get('Odor Data/c')[time+62, xlims, ylims].T
    odor_gradient = f2.get('Odor Data/c_grad_spatial')[ylims, xlims, time+62]
    odor_grad_x = f2.get('Odor Data/c_grad_x')[ylims, xlims, time+62]
    odor_grad_y = f2.get('Odor Data/c_grad_y')[ylims, xlims, time+62]
    # x and y grids for plotting
    x_grid = f2.get(f'Model Metadata/xGrid')[xlims, ylims].T
    y_grid = f2.get(f'Model Metadata/yGrid')[xlims, ylims].T

### SECTION FOR GENERATING ODOR GRADIENT DATASET ####
# initialize new dataset with dimensions
# with h5py.File(file2, 'r+') as f2:
#     f2.create_dataset('Odor Data/c_grad_x', (1201, 1501, 9000), dtype='f4')
#     f2.create_dataset('Odor Data/c_grad_y', (1201, 1501, 9000), dtype='f4')

# chunk_size = 500
# start_idxs = list(range(10, 18))
# start_idxs = [i * chunk_size for i in start_idxs]

# for start_idx in start_idxs:
#     with h5py.File(file2, 'r+') as f2:
#         odor = f2.get('Odor Data/c')[start_idx:start_idx+chunk_size, :, :].T

#         odor = np.flip(odor, axis=0)

#         # Compute spatial concentration gradient as max of dC/dx or dC/dy
#         grad_y, grad_x = np.gradient(odor, axis=(0, 1))
#         # tot_gradient = np.sqrt(grad_x**2 + grad_y**2)
#         # grad_angles = np.arctan2(grad_y, grad_x) * 180/ np.pi

#         # store gradient array in h5 file
#         # gradient_data = f2['Odor Data/c_grad_spatial']
#         # gradient_data[:, :, start_idx:start_idx+chunk_size] = tot_gradient
#         grad_x_data = f2['Odor Data/c_grad_x']
#         grad_x_data[:, :, start_idx:start_idx+chunk_size] = grad_x
#         grad_y_data = f2['Odor Data/c_grad_y']
#         grad_y_data[:, :, start_idx:start_idx+chunk_size] = grad_y

#         print(f'start idx {start_idx} complete.')
### END ODOR GRADIENT SECTION ###


# # Compute spatial FTLE gradient as max of d(FTLE)/dx or d(FTLE)/dy
# # ftle_grad_y, ftle_grad_x = np.gradient(ftle)
# # ftle_grad_x = abs(ftle_grad_x)
# # ftle_grad_y = abs(ftle_grad_y)
# # ftle_max_gradient = np.maximum(ftle_grad_x, ftle_grad_y)

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
cbar1 = plt.colorbar()
cbar1.set_label('FTLE')

# Plot concentration gradient results
# odor[np.where(odor<1*0.0001)] = 1*0.0001
vmin = 1*0.0001
vmax = 0.5
colormap = plt.cm.Reds
colormap.set_under('white')
# plt.pcolormesh(x_grid[:, :], y_grid[:, :], np.flipud(odor[:, :]), cmap=plt.cm.Reds, norm=colors.LogNorm(vmin=None, vmax=None), alpha=0.5)
plt.pcolormesh(x_grid[:, :], y_grid[:, :], odor_gradient[:-1, :-1], cmap=colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
cbar2 = plt.colorbar()
cbar2.set_label('Concentration gradient (mg/m3 per m)')

# Plot arrows for odor gradient direction
# identify local max gradient in coarse grid
n = 5
ny = 5
plt.quiver(x_grid[::ny, ::n], y_grid[::ny, ::n], -odor_grad_x[::ny, ::n], -odor_grad_y[::ny, ::n])
# plt.quiver(x_grid[:], y_grid[:], odor_grad_x[:], odor_grad_y[:], headwidth=6, headlength=7)

# # # Plot velocity magnitude & arrows
# # # cmap = cmr.waterlily_r
# # # plt.pcolormesh(x_grid[:, :], y_grid[:, :], u_tot[:, :], cmap=cmap, vmin=-0.25, vmax=0.25)
# # # plt.colorbar()
# # # n = 150
# # # ny = 100
# # # plt.quiver(x_grid[::ny, ::n], y_grid[::ny, ::n], u[::ny, ::n], v[::ny, ::n], headwidth=6, headlength=7)


# # # Plot max principal strain
# # # vmin = 0.5
# # # vmax = 10
# # # # plt.pcolormesh(x_grid[:, :], y_grid[:, :], strain[:, :], cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.5)
# # # plt.pcolormesh(x_grid[:, :], y_grid[:, :], strain[:, :], cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
# # # plt.colorbar()

ax.set_aspect('equal', adjustable='box')
plt.xlabel('x [meters]')
plt.ylabel('y [meters]')
plt.title(f'FTLE vs odor gradient, t={round((time+62)*0.02, 1)}')

# # plt.savefig('C:/Users/elles/Documents/CU_Boulder/Fluids_Research/Whiffs/ignore/plots/ftle_odorgrad_extended_t287.png', dpi=1000)
plt.show()

