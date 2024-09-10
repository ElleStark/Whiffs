# Source code to run on CU Boulder HPC resources: Blanca cluster
# Computes integral length scales for multisource plume dataset for u and v in both x-stream and streamwise directions
# v2 uses distributed data loading to load chunks of u_data into different processes
# Elle Stark May 2024

import h5py
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import numpy.linalg as LA
from math import log, sqrt
from scipy.interpolate import RegularGridInterpolator
import time
# import os


# Set up logging for convenient messages
logger = logging.getLogger('ftlempipy')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warn
DEBUG = logger.debug


def load_data_chunk(filename, dataset_name, start_idx=None, end_idx=None, ndims=3):
    
    with h5py.File(filename, 'r') as f:
        if ndims==3:
            data_chunk = f.get(dataset_name)[start_idx:end_idx, :, :].astype(np.float64)
            data_chunk = data_chunk.transpose(2, 1, 0)
        elif ndims==2:
            data_chunk = f.get(dataset_name)[:].astype(np.float64)
            data_chunk = data_chunk.T 
        elif ndims==1:
            data_chunk = f.get(dataset_name)[:].astype(np.float64)
        elif ndims==0:
            data_chunk = f.get(dataset_name)[0].item()
        else:
            print('Cannot process number of dimensions. Options are 0, 1, 2, or 3.')

    return data_chunk


def get_vfield(filename, t0, y, dt, xmesh, ymesh):
    # load u and v data from PetaLibrary
    u_dataset_name = 'Flow Data/u' 
    v_dataset_name = 'Flow Data/v'

    # Convert from time to frame
    frame = int(t0 / dt)
    u_data = load_data_chunk(filename, u_dataset_name, frame, frame+1)
    v_data = load_data_chunk(filename, v_dataset_name, frame, frame+1)
    u_data = np.squeeze(u_data)
    v_data = np.squeeze(v_data)
    # with h5py.File(filename, 'r') as f:
    #     u_data = f.get(u_dataset_name)[frame, :, :].astype(np.float64)
    #     u_data = u_data.T
    #     v_data = f.get(v_dataset_name)[frame, :, :].astype(np.float64)
    #     v_data = v_data.T

    ymesh_vec = np.flipud(ymesh)[:, 0]
    xmesh_vec = xmesh[0, :]
    # ymesh_vec = np.flipud(load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2))[:, 0]
    # xmesh_vec = load_data_chunk(filename, 'Model Metadata/xGrid', ndims=2)[0, :]

    x_grid = xmesh
    x_offset = xmesh_vec[-1] / 2
    x_grid = x_grid - x_offset  # Center x coordinates on zero for velocity field extension
    y_grid = ymesh

    # Set up interpolation functions
    # can use cubic interpolation for continuity between the segments (improve smoothness)
    # set bounds_error=False to allow particles to go outside the domain by extrapolation
    u_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(u_data)),
                                        method='linear', bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(v_data)),
                                        method='linear', bounds_error=False, fill_value=None)
    
    # Shift grid such that x is centered on 0 as well as y
    xmesh_vec = xmesh_vec - x_offset
    
    # Define length of transition zone and boundaries
    Delta = 0.0005 * 4
    x_max = np.max(x_grid)
    y_max = np.max(y_grid)

    # Spatial average of velocity field over grid for this time
    avg_u = np.mean(u_data)
    avg_v = np.mean(v_data)
    
    # Tensor for linear velocity field computation
    v_l_tensor = np.empty((2, 2), dtype=np.float32)
    v_l_tensor[0, 0] = np.mean(x_grid * u_data - y_grid * v_data) / np.mean(x_grid ** 2 + y_grid ** 2)
    v_l_tensor[0, 1] = np.mean(y_grid * u_data) / np.mean(y_grid ** 2)
    v_l_tensor[1, 0] = np.mean(x_grid * v_data) / np.mean(x_grid ** 2)
    v_l_tensor[1, 1] = np.mean(y_grid * v_data - x_grid * u_data) / np.mean(x_grid ** 2 + y_grid ** 2)
    
    # Get x and y points
    x_pts = y[0, :] - x_offset
    y_pts = y[1, :]

    # Calculate distance from center for x and y
    x_abs = np.abs(x_pts)
    y_abs = np.abs(y_pts)

    # Outside the grid condition
    outside_cond = (x_abs >= x_max) | (y_abs >= y_max)

    # Inside grid condition
    inside_cond = (x_abs <= (x_max - Delta)) & (y_abs <= (y_max - Delta))

    # Transition zone condition
    transition_cond = ~outside_cond & ~inside_cond

    # Interpolate u and v for points inside the grid
    u_inside = u_interp((y_pts, x_pts + x_offset))
    v_inside = v_interp((y_pts, x_pts + x_offset))

    # For points outside, use linear extension of velocity field
    u_outside = v_l_tensor[0, 0] * x_pts + v_l_tensor[0, 1] * y_pts + avg_u
    v_outside = v_l_tensor[1, 0] * x_pts + v_l_tensor[1, 1] * y_pts + avg_v

    # Delta functions for transition zone
    delta_x = np.where(x_abs <= (x_max - Delta), Delta ** 3,
                    2 * x_abs ** 3 + 3 * (Delta - 2 * x_max) * x_abs ** 2 + 6 * x_max * (x_max - Delta) * x_abs + x_max ** 2 * (3 * Delta - 2 * x_max))
    delta_y = np.where(y_abs <= (y_max - Delta), Delta ** 3,
                    2 * y_abs ** 3 + 3 * (Delta - 2 * y_max) * y_abs ** 2 + 6 * y_max * (y_max - Delta) * y_abs + y_max ** 2 * (3 * Delta - 2 * y_max))

    # Compute velocity for transition zone
    v_l_u = v_l_tensor[0, 0] * x_pts + v_l_tensor[0, 1] * y_pts + avg_u
    v_l_v = v_l_tensor[1, 0] * x_pts + v_l_tensor[1, 1] * y_pts + avg_v

    u_orig = u_interp((y_pts, x_pts + x_offset))
    v_orig = v_interp((y_pts, x_pts + x_offset))

    u_transition = v_l_u + (u_orig - v_l_u) * delta_x * delta_y / Delta ** 6
    v_transition = v_l_v + (v_orig - v_l_v) * delta_x * delta_y / Delta ** 6

    # Combine results using np.where
    u = np.where(outside_cond, u_outside, np.where(inside_cond, u_inside, u_transition))
    v = np.where(outside_cond, v_outside, np.where(inside_cond, v_inside, v_transition))

    vfield = np.array([u, v])

    # plot_times = [0, 10, 20, 30, 40, 50, 60]

    # if t0 in plot_times:
    #     # Plot u: horizontal component of velocity
    #     plt.pcolormesh(u.reshape(2402, 3002))
    #     plt.savefig(f'/rc_scratch/elst4602/LCS_project/QC_plots/u_interp_t{t0}.png')

    #     # Plot v: vertical component of velocity
    #     plt.pcolormesh(v.reshape(2402, 3002))
    #     plt.savefig(f'/rc_scratch/elst4602/LCS_project/QC_plots/v_interp_t{t0}.png')

    return vfield


def advect_improvedEuler(filename, t0, y0, dt, ftle_dt, xmesh, ymesh):
    # get the slopes at the initial and end points
    f1 = get_vfield(filename, t0, y0, dt, xmesh, ymesh)
    f2 = get_vfield(filename, t0 + ftle_dt, y0 + ftle_dt * f1, dt, xmesh, ymesh)
    y_out = y0 + ftle_dt / 2 * (f1 + f2)

    return y_out 


def find_max_eigval(A):
    """The function computes the eigenvalues and eigenvectors of a two-dimensional symmetric matrix.
    from TBarrier repository by Encinas Bartos, Kaszas, Haller 2023: https://github.com/EncinasBartos/TBarrier
    Parameters:
        A: array(2,2), input matrix


    Returns:
        lambda_min: float, minimal eigenvalue
        lambda_max: float, maximal eigenvalue
        v_min: array(2,), minimal eigenvector
        v_max: array(2,), maximal eigenvector
    """
    A11 = A[0, 0]  # float
    A12 = A[0, 1]  # float
    A22 = A[1, 1]  # float

    discriminant = (A11 + A22) ** 2 / 4 - (A11 * A22 - A12 ** 2)  # float

    if discriminant < 0 or np.isnan(discriminant):
        return np.nan, np.nan, np.zeros((1, 2)) * np.nan, np.zeros((1, 2)) * np.nan

    lambda_max = (A11 + A22) / 2 + np.sqrt(discriminant)  # float
    # lambda_min = (A11 + A22) / 2 - np.sqrt(discriminant)  # float

    # v_max = np.array([-A12, A11 - lambda_max])  # array (2,)
    # v_max = v_max / np.sqrt(v_max[0] ** 2 + v_max[1] ** 2)  # array (2,)

    # v_min = np.array([-v_max[1], v_max[0]])  # array (2,)

    return lambda_max


def compute_flow_map(filename, start_t, integration_t, dt, ftle_dt, nx, ny, xmesh_ftle, ymesh_ftle, xmesh_uv, ymesh_uv):
    
    n_steps = abs(int(integration_t / ftle_dt))  # number of timesteps in integration time
    if start_t == 0:
        DEBUG(f'Timesteps in integration time: {n_steps}.')
    
    # Set up initial conditions
    yIC = np.zeros((2, nx * ny))
    yIC[0, :] = xmesh_ftle.reshape(nx * ny)
    yIC[1, :] = ymesh_ftle.reshape(nx * ny)

    y_in = yIC

    for step in range(n_steps):
        tstep = step * ftle_dt + start_t
        y_out = advect_improvedEuler(filename, tstep, y_in, dt, ftle_dt, xmesh_uv, ymesh_uv)
        y_in = y_out

    y_out = np.squeeze(y_out)

    return y_out


def compute_ftle(filename, xmesh_ftle, ymesh_ftle, start_t, integration_t, dt, ftle_dt, spatial_res, xmesh_uv, ymesh_uv):
    # Extract grid dimensions
    grid_height = len(ymesh_ftle[:, 0])
    grid_width = len(xmesh_ftle[0, :])
    
    # Compute flow map (final positions of particles - initial positions already stored in mesh_ftle arrays)
    final_pos = compute_flow_map(filename, start_t, integration_t, dt, ftle_dt, grid_width, grid_height, xmesh_ftle, ymesh_ftle, xmesh_uv, ymesh_uv)
    x_final = final_pos[0]
    x_final = x_final.reshape(grid_height, grid_width)
    y_final = final_pos[1]
    y_final = y_final.reshape(grid_height, grid_width)

    # Initialize arrays for jacobian approximation and ftle
    jacobian = np.empty([2, 2], float)
    ftle = np.zeros([grid_height, grid_width], float)

    # Loop through positions and calculate ftle at each point
    # Leave borders equal to zero (central differencing needs adjacent points for calculation)
    for i in range(1, grid_width - 1):
        for j in range(1, grid_height - 1):
            jacobian[0][0] = (x_final[j, i + 1] - x_final[j, i - 1]) / (2 * spatial_res)
            jacobian[0][1] = (x_final[j + 1, i] - x_final[j - 1, i]) / (2 * spatial_res)
            jacobian[1][0] = (y_final[j, i + 1] - y_final[j, i - 1]) / (2 * spatial_res)
            jacobian[1][1] = (y_final[j + 1, i] - y_final[j - 1, i]) / (2 * spatial_res)

            # Cauchy-Green tensor
            gc_tensor = np.dot(np.transpose(jacobian), jacobian)
    
            # compute eigenvalues and eigenvectors of CG tensor
            # max_eig = find_max_eigval(gc_tensor)  # minval, maxval, vecs

            # its largest eigenvalue
            lamda = LA.eigvals(gc_tensor)
            max_eig = np.max(lamda)

            # Compute FTLE at each location
            ftle[j][i] = 1 / (abs(integration_t)) * log(sqrt(abs(max_eig)))

    return ftle


def plot_ftle_snapshot(ftle_field, xmesh, ymesh, odor=False, fname=None, frame=None, odor_xmesh=None, odor_ymesh=None):
    fig, ax = plt.subplots()

    ftle_field = np.squeeze(ftle_field[0, :, :])

    # Get desired FTLE snapshot data
    plt.contourf(xmesh, ymesh, ftle_field, 100, cmap=plt.cm.Greys, vmin=0, vmax=8)
    plt.title('Odor (red) overlaying FTLE (gray lines)')
    plt.colorbar()

    ax.set_aspect('equal', adjustable='box')
    
    # overlay odor data if desired
    if odor:
        odor_data = load_data_chunk(fname, 'Odor Data/c', frame, frame+1, ndims=3)
        odor_data = np.squeeze(odor_data)
        plt.pcolormesh(odor_xmesh, odor_ymesh, odor_data, cmap=plt.cm.Reds, alpha=0.5, vmax=0.5)
        plt.colorbar()
        ax.set_aspect('equal', adjustable='box')

    return fig


def main():

    # MPI setup and related data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # number of processes will be determined from ntasks listed on Slurm job script (.sh file) 
    num_procs = comm.Get_size()
    # INFO(f'RUNNING ON {num_procs} PROCESSES.')

    # Define common variables on all processes
    filename = '/pl/active/odor2action/Stark_data/Re100_0_5mm_50Hz_singlesource_2d.h5'
    # scratch_dir = os.environ['SLURM_SCRATCH']
    # filename = f'{scratch_dir}/Re100_0_5mm_50Hz_singlesource_2d.h5'
    integration_time = 0.6  # seconds

    # Define dataset-based variables on process 0
    if rank==0:
        DEBUG("starting Python script")
        start = time.time()
        spatial_res = load_data_chunk(filename, 'Model Metadata/spatialResolution', ndims=0)
        dt_freq = load_data_chunk(filename, 'Model Metadata/timeResolution', ndims=0)
        dt = 1 / dt_freq  # convert from Hz to seconds; negative for backward-time FTLE
        # ymesh_uv = np.flipud(load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2))
        ymesh_uv = load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2)
        xmesh_uv = load_data_chunk(filename, 'Model Metadata/xGrid', ndims=2)
        u_grid_dims = np.shape(xmesh_uv)
        x_min = xmesh_uv[0][0]
        x_max = xmesh_uv[0][-1]
        y_min = ymesh_uv[0][0]
        y_max = ymesh_uv[-1][0]
        # ymesh_uv = np.ascontiguousarray(ymesh_uv)
        # xmesh_uv = np.ascontiguousarray(xmesh_uv)
        ymesh_uv = ymesh_uv.flatten()
        xmesh_uv = xmesh_uv.flatten()
        duration = load_data_chunk(filename, 'Model Metadata/datasetDuration', ndims=0)
        duration = (duration - integration_time) / dt  # adjust duration to account for advection time at first FTLE step

        # Create grid of particles with desired spacing
        particle_spacing = spatial_res / 2  # can determine visually if dx is appropriate based on smooth contours for FTLE

        # x and y vectors based on velocity mesh limits and particle spacing
        xmesh_ftle = np.linspace(x_min, x_max, int(u_grid_dims[1] * spatial_res/particle_spacing))
        ymesh_ftle = np.linspace(y_min, y_max, int(u_grid_dims[0] * spatial_res/particle_spacing))
        xmesh_ftle, ymesh_ftle = np.meshgrid(xmesh_ftle, ymesh_ftle, indexing='xy')
        # xmesh_ftle = np.ascontiguousarray(xmesh_ftle)
        ymesh_ftle = np.flipud(ymesh_ftle)
        # ymesh_ftle = np.ascontiguousarray(ymesh_ftle)
        grid_dims = np.shape(xmesh_ftle)
        ymesh_ftle = ymesh_ftle.flatten()
        xmesh_ftle = xmesh_ftle.flatten()
        DEBUG(f'Grid dimensions: {grid_dims}.')
        DEBUG(f"Time to load metadata on process 0: {time.time() - start} s.")

    else:
        # These variables will be broadcast from process 0 based on file contents
        grid_dims = None
        u_grid_dims = None
        dt = None
        duration = None  # total timesteps (idxs) for FTLE calcs
        particle_spacing = None
        spatial_res = None

    # Broadcast dimensions of x and y grids to each process for pre-allocating arrays  
    grid_dims = comm.bcast(grid_dims, root=0) # note, use bcast for Python objects, Bcast for Numpy arrays
    # DEBUG(f'process {rank} grid dims: {grid_dims}.')
    u_grid_dims = comm.bcast(u_grid_dims, root=0)
    dt = comm.bcast(dt, root=0)
    ftle_dt = -dt
    particle_spacing = comm.bcast(particle_spacing, root=0) 
    duration = comm.bcast(duration, root=0) 
    spatial_res = comm.bcast(spatial_res, root=0)
    
    if rank != 0:
        xmesh_ftle = np.empty([grid_dims[0]*grid_dims[1]], dtype='d')
        ymesh_ftle = np.empty([grid_dims[0]*grid_dims[1]], dtype='d')
        xmesh_uv = np.empty([u_grid_dims[0]*u_grid_dims[1]], dtype='d')
        ymesh_uv = np.empty([u_grid_dims[0]*u_grid_dims[1]], dtype='d')
        # xmesh_ftle = np.zeros(grid_dims, dtype='d')
        # ymesh_ftle = np.zeros(grid_dims, dtype='d')
        # xmesh_uv = np.zeros(u_grid_dims, dtype='d')
        # ymesh_uv = np.zeros(u_grid_dims, dtype='d')
        # DEBUG(f'process {rank} mesh buffers set.')

    comm.Bcast([xmesh_ftle, MPI.DOUBLE], root=0)
    comm.Bcast([ymesh_ftle, MPI.DOUBLE], root=0)
    comm.Bcast([xmesh_uv, MPI.DOUBLE], root=0)
    comm.Bcast([ymesh_uv, MPI.DOUBLE], root=0)

    # Reshape x and y meshes
    xmesh_ftle = xmesh_ftle.reshape(grid_dims)
    ymesh_ftle = ymesh_ftle.reshape(grid_dims)
    xmesh_uv = xmesh_uv.reshape(u_grid_dims)
    ymesh_uv = ymesh_uv.reshape(u_grid_dims)

    # Compute chunk sizes for each process - if 180 processes, chunk size should be 49 to 50
    chunk_size = duration // num_procs
    # DEBUG(f'Chunk size: {chunk_size}')
    remainder = duration % num_procs

    # Find start and end time index for each process
    start_idx = int(integration_time / abs(ftle_dt) + rank * chunk_size + min(rank, remainder)) 
    end_idx = int(start_idx + chunk_size + (1 if rank < remainder else 0))
    # DEBUG(f'Process {rank} start idx: {start_idx}; end idx: {end_idx}')

    # if ftle_dt < 0:
    #     # If calculating backward FTLE, need to start at least one integration time after beginning of data
    #     start_idx = start_idx - integration_time / ftle_dt
    #     end_idx = int(start_idx + chunk_size + (1 if rank < remainder else 0))
    # if ftle_dt > 0:
    #     # If calculating forward FTLE, need to end at least one integration time before end of data
    #     start_idx = start_idx
    #     end_idx = end_idx - integration_time / dt

    # Compute FTLE and save to .npy on each process for each timestep
    ftle_chunk = np.zeros([(end_idx - start_idx), grid_dims[0], grid_dims[1]], dtype='d')
    timesteps = range(end_idx - start_idx)
    # timesteps = [0]  # TEST WITH SINGLE TIMESTEP
    # ftle_chunk = np.zeros([1, grid_dims[0], grid_dims[1]], dtype='d')
    ftle_chunk = np.zeros([len(timesteps), grid_dims[0], grid_dims[1]], dtype='d')

    start = time.time()
    DEBUG(f'Began FTLE computation on process {rank} at {start}.')
    for idx in timesteps:
        start_t = (start_idx + idx) * dt
        ftle_field = compute_ftle(filename, xmesh_ftle, ymesh_ftle, start_t, integration_time, dt, ftle_dt, spatial_res, xmesh_uv, ymesh_uv)
        ftle_chunk[idx, :, :] = ftle_field
    DEBUG(f'Ended FTLE computation on process {rank} after {(time.time()-start)/60} min.')

    # dynamic file name in /rc_scratch based on rank/idxs
    data_fname = f'/rc_scratch/elst4602/LCS_project/ftle_data/{rank : 04d}_t{round(start_idx*dt, 2)}to{round(end_idx*dt, 2)}s_singlesource_cylarray_0to180s_ftle.npy'
    np.save(data_fname, ftle_chunk)
    
    # Plot and save figure at final timestep of each process in /rc_scratch
    ftle_fig = plot_ftle_snapshot(ftle_chunk, xmesh_ftle, ymesh_ftle, odor=True, fname=filename, frame=start_idx, odor_xmesh=xmesh_uv, odor_ymesh=ymesh_uv)
    plot_fname = f'/rc_scratch/elst4602/LCS_project/ftle_plots/{rank : 04d}_t{round(start_idx*dt, 2)}s_singlesource_cylarray_0to180s_ftle.png'
    ftle_fig.savefig(plot_fname, dpi=300)

    DEBUG(f"Process {rank} completed with result size {ftle_chunk.shape}")

if __name__=="__main__":
    main()


