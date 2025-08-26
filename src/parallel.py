"""Parallel matrix-vector multiplication

Computes

    C = A @ B

in parallel, where the shapes of the matrices are:

    * A: (n,m)
    * B: (m,r)
    * C: (n,r)

n and m have to be multiples of the number of processors.

Supports overlapping of computation and communications.


"""

import argparse
from mpi4py import MPI
import numpy as np


def parallel_matmul(A, B, C, overlap=False, nocomm=False):
    """Parallel matrix-vector product

    Compute local part of matrix C = A @ B, where each processor only stores its
    local part of the matrices A and B.

    :arg A: process-local part of matrix A
    :arg B: process-local part of matrix B
    :arg C: process-local part of resulting matrix C
    :arg overlap: overlap computation and communication?

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    m_loc, _ = B.shape

    B_recv = np.empty_like(B)
    req_send = None
    req_recv = None
    for q in range(nproc):
        if not nocomm and overlap and q < nproc - 1:
            req_send = comm.Isend(B, dest=(rank - 1) % nproc)
            req_recv = comm.Irecv(B_recv)
        C[:, :] += (
            A[:, ((q + rank) % nproc) * m_loc : (((q + rank) % nproc + 1)) * m_loc]
            @ B[:, :]
        )

        if q < nproc - 1 and not nocomm:
            if overlap:
                req_send.wait()
                req_recv.wait()
                B[:, :] = B_recv[:, :]
            else:
                comm.Sendrecv_replace(B, (rank - 1) % nproc)
    return C


parser = argparse.ArgumentParser()

parser.add_argument(
    "--n",
    action="store",
    default=128,
    required=True,
    type=int,
    help="Number of rows of matrix A",
)

parser.add_argument(
    "--m",
    action="store",
    required=False,
    type=int,
    help="Number of columns of matrix A",
)

parser.add_argument(
    "--r",
    action="store",
    required=False,
    type=int,
    help="Number of columns of matrix B",
)

parser.add_argument(
    "--overlap",
    action="store_true",
    help="Overlap calculation and communication?",
)

parser.add_argument(
    "--nocomm",
    action="store_true",
    help="No parallel communication (leads to incorrect results)?",
)


parser.add_argument(
    "--niter",
    action="store",
    default=10,
    type=int,
    help="Number repetitions",
)


args, _ = parser.parse_known_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

n = args.n
m = args.m if args.m else args.n
r = args.r if args.r else args.n

assert n % nproc == 0, "n must be a multiple of the number of processors"
assert m % nproc == 0, "m must be a multiple of the number of processors"

n_loc = n // nproc
m_loc = m // nproc

if rank == 0:
    print(f"Running on {nproc} processors.")
    print(f" n = {n:8d}")
    print(f" m = {m:8d}")
    print(f" r = {r:8d}")
    print(f" overlap = {args.overlap}")
    print(f" nocomm  = {args.nocomm}")

rng = np.random.default_rng(seed=411587)
# Global matrices
A = rng.normal(size=(n, m))
B = rng.normal(size=(m, r))
# Result for testing
C_true = A @ B
t_elapsed = 0
for _ in range(args.niter):
    A_loc = np.array(A[rank * n_loc : (rank + 1) * n_loc, :])
    B_loc = np.array(B[rank * m_loc : (rank + 1) * m_loc, :])
    C_loc = np.zeros(shape=(n_loc, r))
    comm.Barrier()
    t_start = MPI.Wtime()
    parallel_matmul(A_loc, B_loc, C_loc, args.overlap, args.nocomm)
    comm.Barrier()
    t_finish = MPI.Wtime()
    C = np.zeros(shape=(n, r))
    comm.Allgather(C_loc, C)
    t_elapsed += (t_finish - t_start) / args.niter
    # t_elapsed = min(t_finish - t_start, t_elapsed)

if rank == 0:
    nrm = np.linalg.norm(C - C_true) / np.linalg.norm(C_true)
    print(f"error norm = {nrm:8.4e}")

    print(f"elapsed time = {t_elapsed:8.2e} s")
