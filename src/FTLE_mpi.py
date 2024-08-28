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

    ymesh_vec = ymesh[:, 0]
    xmesh_vec = xmesh[0, :]
    # ymesh_vec = np.flipud(load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2))[:, 0]
    # xmesh_vec = load_data_chunk(filename, 'Model Metadata/xGrid', ndims=2)[0, :]

    # Set up interpolation functions
    # can use cubic interpolation for continuity of the between the segments (improve smoothness)
    # set bounds_error=False to allow particles to go outside the domain by extrapolation
    u_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(u_data)),
                                        method='linear', bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(v_data)),
                                        method='linear', bounds_error=False, fill_value=None)
    
    u = u_interp((y[1], y[0]))
    v = v_interp((y[1], y[0]))
    vfield = np.array([u, v])

    return vfield


def advect_improvedEuler(filename, t0, y0, dt, xmesh, ymesh):
    # get the slopes at the initial and end points
    f1 = get_vfield(filename, t0, y0, dt, xmesh, ymesh)
    f2 = get_vfield(filename, t0 + dt, y0 + dt * f1, dt, xmesh, ymesh)
    y_out = y0 + dt / 2 * (f1 + f2)

    return y_out 


def compute_flow_map(filename, start_t, integration_t, dt, nx, ny, xmesh_ftle, ymesh_ftle):
    
    n_steps = abs(int(integration_t / dt))  # number of timesteps in integration time
    if start_t == 0:
        DEBUG(f'Timesteps in integration time: {n_steps}.')
    
    # Set up initial conditions
    yIC = np.zeros((2, nx * ny))
    yIC[0, :] = xmesh_ftle.reshape(nx * ny)
    yIC[1, :] = ymesh_ftle.reshape(nx * ny)

    y_in = yIC

    for step in range(n_steps):
        tstep = step * dt + start_t
        y_out = advect_improvedEuler(filename, tstep, y_in, dt, xmesh_ftle, ymesh_ftle)
        y_in = y_out

    y_out = np.squeeze(y_out)

    return y_out


def compute_ftle(filename, xmesh_ftle, ymesh_ftle, start_t, integration_t, dt, spatial_res):
    # Extract grid dimensions
    grid_height = len(ymesh_ftle[:, 0])
    grid_width = len(xmesh_ftle[0, :])
    
    # Compute flow map (final positions of particles - initial positions already stored in mesh_ftle arrays)
    final_pos = compute_flow_map(filename, start_t, integration_t, dt, grid_width, grid_height, xmesh_ftle, ymesh_ftle)
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
    
            # its largest eigenvalue
            lamda = LA.eigvals(gc_tensor)
            max_eig = np.max(lamda)

            # Compute FTLE at each location
            ftle[j][i] = 1 / (abs(integration_t)) * log(sqrt(max_eig))

    return ftle


def plot_ftle_snapshot(ftle_field, xmesh, ymesh, odor=False, fname=None, frame=None):
    fig, ax = plt.subplots()

    ftle_field = np.squeeze(ftle_field[-1:, :, :])

    # Get desired FTLE snapshot data
    plt.contourf(xmesh, ymesh, ftle_field, 100, cmap=plt.cm.Greys)
    plt.title('Odor (red) overlaying FTLE (gray lines)')
    plt.colorbar()

    ax.set_aspect('equal', adjustable='box')
    
    # overlay odor data if desired
    if odor:
        odor_data = load_data_chunk(fname, 'Odor Data/c', frame, frame+1, ndims=3)
        plt.pcolormesh(xmesh, ymesh, odor_data, cmap=plt.cm.Reds, alpha=0.5, vmax=0.5)
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
    integration_time = 0.6  # seconds

    # These variables will be broadcast from process 0 based on file contents
    grid_dims = None
    dt = None
    duration = None  # total timesteps (idxs) for FTLE calcs
    particle_spacing = None

    # Define dataset-based variables on process 0
    if rank==0:
        
        spatial_res = load_data_chunk(filename, 'Model Metadata/spatialResolution', ndims=0)
        dt_freq = load_data_chunk(filename, 'Model Metadata/timeResolution', ndims=0)
        dt = 1 / dt_freq  # convert from Hz to seconds
        ymesh_uv = np.flipud(load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2))
        xmesh_uv = load_data_chunk(filename, 'Model Metadata/xGrid', ndims=2)
        duration = len(load_data_chunk(filename, 'Model Metadata/timeArray', ndims=1))
        duration = duration - integration_time / dt  # adjust duration to account for advection time at last FTLE step

        # Create grid of particles with desired spacing
        particle_spacing = spatial_res / 2  # can determine visually if dx is appropriate based on smooth contours for FTLE

        # x and y vectors based on velocity mesh limits and particle spacing
        xvec_ftle = np.linspace(xmesh_uv[0][0], xmesh_uv[0][-1], int(np.shape(xmesh_uv)[1] * spatial_res/particle_spacing))
        yvec_ftle = np.linspace(ymesh_uv[0][0], ymesh_uv[-1][0], int(np.shape(xmesh_uv)[0] * spatial_res/particle_spacing))
        xmesh_ftle, ymesh_ftle = np.meshgrid(xvec_ftle, yvec_ftle, indexing='xy')
        ymesh_ftle = np.flipud(ymesh_ftle)
        grid_dims = xmesh_ftle.shape

    # Broadcast dimensions of x and y grids to each process for pre-allocating arrays  
    comm.bcast(grid_dims, root=0) # note, use bcast for Python objects, Bcast for Numpy arrays
    comm.bcast(dt, root=0)
    comm.bcast(particle_spacing, root=0) 
    comm.bcast(duration, root=0) 
    
    if rank != 0:
        xmesh_ftle = np.empty(grid_dims, dtype='d')
        ymesh_ftle = np.empty(grid_dims, dtype='d')

    comm.Bcast([xmesh_ftle, MPI.DOUBLE], root=0)
    comm.Bcast([ymesh_ftle, MPI.DOUBLE], root=0)

    # Compute chunk sizes for each process - if 180 processes, chunk size should be 49 to 50
    chunk_size = duration // num_procs
    DEBUG(f'Chunk size: {chunk_size}')
    remainder = duration % num_procs

    # Find start and end time index for each process
    start_idx = rank * chunk_size + min(rank, remainder) 
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0) 
    DEBUG(f'start idx: {start_idx}; end idx: {end_idx}')


    # Compute FTLE and save to .npy on each process for each timestep
    ftle_chunk = np.zeros([(end_idx - start_idx), grid_dims[0], grid_dims[1]], dtype='d')
    for idx in range(end_idx - start_idx):
        start_t = (start_idx + idx) * dt
        ftle_field = compute_ftle(filename, xmesh_ftle, ymesh_ftle, start_t, integration_time, dt, spatial_res)
        ftle_chunk[idx, :, :] = ftle_field

    # dynamic file name in /rc_scratch based on rank/idxs
    data_fname = f'/rc_scratch/elst4602/LCS_project/ftle_data/{rank}_t{start_idx*dt}to{end_idx*dt}s_singlesource_cylarray_0to180s_ftle.npy'
    np.save(data_fname, ftle_chunk)
    
    # Plot and save figure at final timestep of each process in /rc_scratch
    ftle_fig = plot_ftle_snapshot(ftle_chunk, xmesh_ftle, ymesh_ftle, odor=True, fname=filename, frame=end_idx)
    plot_fname = f'/rc_scratch/elst4602/LCS_project/ftle_plots/{rank}_t{start_idx*dt}to{end_idx*dt}s_singlesource_cylarray_0to180s_ftle.png'
    plt.savefig(plot_fname, ftle_fig, dpi=300)

    DEBUG(f"Process {rank} completed with result size {ftle_chunk.shape}")

if __name__=="__main__":
    main()


