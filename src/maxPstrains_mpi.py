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


def load_data_chunk(filename, dataset_name, start_idx=None, end_idx=None, ndims=3, transpose=True):
    
    with h5py.File(filename, 'r') as f:
        if ndims==3:
            data_chunk = f.get(dataset_name)[start_idx:end_idx, :, :].astype(np.float64)
            if transpose:
                data_chunk = data_chunk.transpose(2, 1, 0)
        elif ndims==2:
            data_chunk = f.get(dataset_name)[:].astype(np.float64)
            if transpose:
                data_chunk = data_chunk.T 
        elif ndims==1:
            data_chunk = f.get(dataset_name)[:].astype(np.float64)
        elif ndims==0:
            data_chunk = f.get(dataset_name)[0].item()
        else:
            print('Cannot process number of dimensions. Options are 0, 1, 2, or 3.')

    return data_chunk


def compute_max_strain(filename, dx, start_idx, end_idx):
    # load u and v data from PetaLibrary (all tsteps)
    u_dataset_name = 'Flow Data/u' 
    v_dataset_name = 'Flow Data/v'
    
    u_data = load_data_chunk(filename, u_dataset_name, start_idx, end_idx, transpose=False)
    v_data = load_data_chunk(filename, v_dataset_name, start_idx, end_idx, transpose=False)
    
    # Flip velocity data upside-down for correct computation of gradients
    u_data = np.flip(u_data, axis=2)
    v_data = np.flip(v_data, axis=2)

    # compute spatial gradients
    dudx = np.gradient(u_data, dx, axis=1)
    dudy = np.gradient(u_data, dx, axis=2)
    dvdx = np.gradient(v_data, dx, axis=1)
    dvdy = np.gradient(v_data, dx, axis=2)

    dims = np.shape(dudx)  # x, y, time
    DEBUG(f'data dimensions are: {dims}')  # confirm dimensions

    # Stack and reshape to gradients to create an array of Jacobian matrices for each point in each timestep
    jacobians = np.stack((dudx, dvdx, dudy, dvdy), axis=-1).reshape(dims[0], dims[1], dims[2], 2, 2)

    # Compute strain tensors
    strain_tensors = 0.5 * (jacobians + np.transpose(jacobians, (0, 1, 2, 4, 3)))

    # Initialize eigenvector and eigenvalue arrays - save max and min separately for clarity in H5 file
    max_eigval_stack = np.zeros((dims[0], dims[1], dims[2]))

    # Numpy's eig function doesn't support vectorization, so loop through all points to get eigenvalues 
    for t in range(dims[0]):
        for i in range(dims[1]):
            for j in range(dims[2]):
                # compute eigenvalues and save max value
                eigvals = np.linalg.eigvals(strain_tensors[t, i, j])
                max_eigval_stack[t, i, j] = np.max(eigvals)

    return max_eigval_stack


def plot_strain_snapshot(strain_field, xmesh, ymesh):
    fig, ax = plt.subplots()

    strain_field = np.squeeze(strain_field[0, :, :]).T

    # Get desired FTLE snapshot data
    plt.pcolormesh(xmesh, ymesh, strain_field[:-1, :-1], cmap=plt.cm.viridis, vmin=0, vmax=18)
    plt.colorbar()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('maximum principal strain')

    ax.set_aspect('equal', adjustable='box')

    return fig


def main():

    # MPI setup and related data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # number of processes will be determined from ntasks listed on Slurm job script (.sh file) 
    num_procs = comm.Get_size()

    # Define common variables on all processes
    filename = '/pl/active/odor2action/Stark_data/Re100_0_5mm_50Hz_singlesource_2d.h5'

    # Define dataset-based variables on process 0
    if rank==0:
        DEBUG("starting Python script")
        start = time.time()
        spatial_res = load_data_chunk(filename, 'Model Metadata/spatialResolution', ndims=0)
        dt_freq = load_data_chunk(filename, 'Model Metadata/timeResolution', ndims=0)
        dt = 1 / dt_freq  # convert from Hz to seconds; negative for backward-time FTLE

        ymesh_uv = load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2)
        xmesh_uv = load_data_chunk(filename, 'Model Metadata/xGrid', ndims=2)
        u_grid_dims = np.shape(xmesh_uv)

        ymesh_uv = ymesh_uv.flatten()
        xmesh_uv = xmesh_uv.flatten()
        duration = load_data_chunk(filename, 'Model Metadata/datasetDuration', ndims=0)
        duration = duration / dt

    else:
        # These variables will be broadcast from process 0 based on file contents
        u_grid_dims = None
        dt = None
        duration = None  # total timesteps (idxs) for FTLE calcs
        spatial_res = None

    # Broadcast dimensions of x and y grids to each process for pre-allocating arrays  
    u_grid_dims = comm.bcast(u_grid_dims, root=0)
    dt = comm.bcast(dt, root=0)
    duration = comm.bcast(duration, root=0) 
    spatial_res = comm.bcast(spatial_res, root=0)
    
    if rank != 0:
        xmesh_uv = np.empty([u_grid_dims[0]*u_grid_dims[1]], dtype='d')
        ymesh_uv = np.empty([u_grid_dims[0]*u_grid_dims[1]], dtype='d')

    comm.Bcast([xmesh_uv, MPI.DOUBLE], root=0)
    comm.Bcast([ymesh_uv, MPI.DOUBLE], root=0)

    # Reshape x and y meshes
    xmesh_uv = xmesh_uv.reshape(u_grid_dims)
    ymesh_uv = ymesh_uv.reshape(u_grid_dims)

    # Compute chunk sizes for each process - if 180 processes, chunk size should be 49 to 50
    chunk_size = duration // num_procs
    remainder = duration % num_procs

    # Find start and end time index for each process
    start_idx = int(rank * chunk_size + min(rank, remainder)) 
    end_idx = int(start_idx + chunk_size + (1 if rank < remainder else 0))

    start = time.time()
    DEBUG(f'Began max P strain computation on process {rank} at {start}.')

    strain_chunk = compute_max_strain(filename, spatial_res, start_idx, end_idx)

    DEBUG(f'Ended strain computation on process {rank} after {(time.time()-start)/60} min.')

    # dynamic file name in /rc_scratch based on rank/idxs
    data_fname = f'/rc_scratch/elst4602/LCS_project/strain_data/{rank : 04d}_t{round(start_idx*dt, 2)}to{round(end_idx*dt, 2)}s_singlesource_cylarray_0to180s_maxPstrain.npy'
    np.save(data_fname, strain_chunk)
    
    # Plot and save figure at final timestep of each process in /rc_scratch
    strain_fig = plot_strain_snapshot(strain_chunk, xmesh_uv, ymesh_uv)
    plot_fname = f'/rc_scratch/elst4602/LCS_project/strain_plots/{rank : 04d}_t{round(start_idx*dt, 2)}s_singlesource_cylarray_0to180s_maxPstrain.png'
    strain_fig.savefig(plot_fname, dpi=300)

    DEBUG(f"Process {rank} completed with result size {strain_chunk.shape}")

if __name__=="__main__":
    main()


