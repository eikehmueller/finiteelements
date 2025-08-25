from mpi4py import MPI
import numpy as np
from fem.utilities import measure_time

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
nproc = comm.Get_size()

n = 256
m = 256
r = 256
n_loc = n // nproc
m_loc = m // nproc

rng = np.random.default_rng(seed=411587)
# Global matrices
A = rng.normal(size=(n, m))
B = rng.normal(size=(m, r))
# Result for testing
C_true = A @ B
niter = 100
overlap = True
t_elapsed = 0
for _ in range(niter):

    A_loc = A[rank * n_loc : (rank + 1) * n_loc, :]
    B_loc = B[rank * m_loc : (rank + 1) * m_loc, :]
    C_loc = np.zeros(shape=(n_loc, r))
    B_recv = np.empty_like(B_loc)

    t_start = MPI.Wtime()
    for q in range(nproc):
        if overlap and q < nproc - 1:
            req_send = comm.Isend(B_loc, dest=(rank - 1) % nproc)
            req_recv = comm.Irecv(B_recv)
        C_loc[:, :] += (
            A_loc[:, ((q + rank) % nproc) * m_loc : (((q + rank) % nproc + 1)) * m_loc]
            @ B_loc[:, :]
        )

        if q < nproc - 1:
            if overlap:
                req_send.wait()
                req_recv.wait()
                B_loc[:, :] = B_recv[:, :]
            else:
                comm.Sendrecv_replace(B_loc, (rank - 1) % nproc)
    t_finish = MPI.Wtime()
    C = np.zeros(shape=(n, r))
    comm.Allgather(C_loc, C)
    t_elapsed += (t_finish - t_start) / niter

if rank == 0:
    nrm = np.linalg.norm(C - C_true) / np.linalg.norm(C_true)
    print(f"error norm = {nrm:8.4e}")

    print(f"elapsed time = {1000*t_elapsed:8.2f} ms")
