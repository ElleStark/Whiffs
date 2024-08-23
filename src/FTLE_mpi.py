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
            data_chunk = f[dataset_name][start_idx:end_idx, :, :].astype(np.float64)
            data_chunk = data_chunk.transpose(2, 1, 0)
        else:
            data_chunk = f[dataset_name][:].astype(np.float64)
            data_chunk = data_chunk.T    

        # DEBUG(f'Memory size of each chunk: {data_chunk.itemsize * data_chunk.size}')
    return data_chunk

def get_vfield(t0, dt, y):
    # load u and v data from PetaLibrary
    filename = '/pl/active/odor2action/Stark_data/Re100_0_5mm_50Hz_singlesource_2d.h5'
    u_dataset_name = 'Flow Data/u' 
    v_dataset_name = 'Flow Data/v'

    # Convert from time to frame
    frame = int(t0 / dt)
    u_data = load_data_chunk(filename, u_dataset_name, frame, frame+1)
    v_data = load_data_chunk(filename, v_dataset_name, frame, frame+1)
    ymesh_vec = np.flipud(load_data_chunk(filename, 'Model Metadata/yGrid', ndims=2))[:, 0]
    xmesh_vec = load_data_chunk(filename, 'Model Metadata/xGrid', ndims=2)[0, :]
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

def advect_improvedEuler(t0, y0, dt):
    # get the slopes at the initial and end points
    f1 = get_vfield(t0, dt, y0)
    f2 = get_vfield(t0 + dt, y0 + dt * f1)
    y_out = y0 + dt / 2 * (f1 + f2)

    return y_out 

def compute_flow_map(integration_t, dt, nx, ny):
    L = abs(int(integration_t / dt))  # need to calculate if dt definition is not based on T
    

def compute_ftle():


def main():

    # MPI setup and related data
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    # number of processes will be determined from ntasks listed on Slurm job script (.sh file) 
    num_procs = comm_world.Get_size()
    INFO(f'RUNNING ON {num_procs} PROCESSES.')

    # Define time, x, and y sizes - hardcoded here for convenience
    t, x, y = 9001, 1501, 1201

    # Compute chunk sizes for each process - if 180 processes, chunk size should be 50
    chunk_size = t // num_procs
    DEBUG(f'Chunk size: {chunk_size}')
    remainder = t % num_procs

    # Find start and end time index for each process
    start_idx = rank * chunk_size + min(rank, remainder) 
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0) 
    DEBUG(f'start idx: {start_idx}; end idx: {end_idx}')

    # Load chunk of data to each process: enough to compute FTLE field for n timesteps, then save to numpy in scratch folder
    local_u_chunk = load_data_chunk(filename, dataset_name, start_idx, end_idx, adjust)
    DEBUG(f'Process {rank} loaded data chunk with shape {local_u_chunk.shape}')

    # Simple QC test for mpi: sum along time axis in each process
    # local_result = np.sum(local_u_chunk, axis=0)

    # LOCAL RESULTS: CALCULATE FINITE TIME LYAPUNOV EXPONENT 
    # Each process computes FTLE field for 50 timesteps and saves to .npy file in /scratch directory
    

    # Save local_ftle to numpy in scratch directory

    DEBUG(f"Process {rank} completed with result size {local_ftle.size}")


    # # GATHER ALL RESULTS INTO PROCESS 0

    # if streamwise:
    #     recvcounts = np.array([chunk_size * x] * num_procs, dtype=int)
    #     for i in range(remainder):
    #         recvcounts[i] += x
    # else:
    #     recvcounts = np.array([chunk_size * y] * num_procs, dtype=int)
    #     for i in range(remainder):
    #         recvcounts[i] += y

    # recvdisplacements = np.array([sum(recvcounts[:i]) for i in range(num_procs)], dtype=int)

    # # Prepare buffer in root process for gathering
    # if rank == 0:
    #     if streamwise:
    #         gathered_u = np.empty((y, x), dtype=np.float64)
    #         # data_test = np.empty((y, x), dtype=np.float64)
    #     else:
    #         gathered_u = np.empty((x, y), dtype=np.float64)
    # else:
    #     gathered_u = None
    #     # data_test = None

    # comm_world.Gatherv(local_result, [gathered_u, recvcounts, recvdisplacements, MPI.DOUBLE], root=0)
    # # comm_world.Gatherv(local_u_mean, [data_test, recvcounts, recvdisplacements, MPI.DOUBLE], root=0)

    # # Reshape gathered data
    # if rank == 0: 
    #     if streamwise:
    #         gathered_u = gathered_u.reshape((y, x))
    #     else:
    #         gathered_u = gathered_u.reshape((x, y)).T
    #     DEBUG(f'gathered data shape: {gathered_u.shape}')
    #     DEBUG(f'y = {y}')
    #     # data_test = data_test.reshape((y, x))

    #     gathered_u = np.flip(gathered_u, axis=0)
    #     INFO("ils array data shape: " + str(gathered_u.shape[0]) + " x " + str(gathered_u.shape[1]))
    #     DEBUG("spatial average across domain: " + str(np.mean(gathered_u)))

    #     if streamwise:
    #         direction = 'streamwise'
    #     else:
    #         direction = 'cross_stream'
    #     np.save(f'/rc_scratch/elst4602/FisherPlumePlots/ILS_{dataset_name}_{direction}.npy', gathered_u)

    #     # QC Plots
    #     fig, ax = plt.subplots(figsize=(5.9, 4))
    #     plt.pcolormesh(gathered_u, cmap='magma_r', vmin=0, vmax=0.20)
    #     plt.colorbar()
    #     plt.savefig(f'/rc_scratch/elst4602/FisherPlumePlots/ILS_{dataset_name}_{direction}.png', dpi=600)

        # plt.close()
        # plt.pcolormesh(data_test)
        # plt.colorbar()
        # plt.savefig('/rc_scratch/elst4602/FisherPlumePlots/uprime0.png', dpi=600)

if __name__=="__main__":
    main()


