from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def counts_displs(M: int, p: int):
    base, rem = divmod(M, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

def main():
    # ====== ПАРАМЕТРЫ ЗАДАЧИ (можешь менять) ======
    # A: M×N, x: длины M, b: длины N
    M = 10000
    N = 200

    # ====== root генерит A и x ======
    if rank == 0:
        rng = np.random.default_rng(123)
        A = rng.random((M, N), dtype=np.float64)
        x = rng.random(M, dtype=np.float64)
        rcounts, displs = counts_displs(M, size)
    else:
        A = None
        x = None
        rcounts = None
        displs = None

    # всем нужны rcounts/displs
    rcounts = comm.bcast(rcounts, root=0)
    displs = comm.bcast(displs, root=0)

    local_M = int(rcounts[rank])

    # ====== Scatterv для A (по строкам) ======
    A_part = np.empty((local_M, N), dtype=np.float64)
    sendcounts_A = (rcounts * N).astype(np.int32)
    displs_A = (displs * N).astype(np.int32)

    if rank == 0:
        comm.Scatterv([A.reshape(-1), sendcounts_A, displs_A, MPI.DOUBLE],
                      [A_part.reshape(-1), local_M * N, MPI.DOUBLE],
                      root=0)
    else:
        comm.Scatterv([None, None, None, None],
                      [A_part.reshape(-1), local_M * N, MPI.DOUBLE],
                      root=0)

    # ====== Scatterv для x (согласованно по строкам) ======
    x_part = np.empty(local_M, dtype=np.float64)
    if rank == 0:
        comm.Scatterv([x, rcounts, displs, MPI.DOUBLE],
                      [x_part, local_M, MPI.DOUBLE],
                      root=0)
    else:
        comm.Scatterv([None, None, None, None],
                      [x_part, local_M, MPI.DOUBLE],
                      root=0)

    # ====== Локальные вычисления ======
    comm.Barrier()
    t0 = MPI.Wtime()
    b_temp = A_part.T @ x_part          # длины N
    comm.Barrier()
    t1 = MPI.Wtime()

    # ====== Reduce (SUM) по вектору длины N ======
    if rank == 0:
        b = np.empty(N, dtype=np.float64)
    else:
        b = None

    comm.Reduce([b_temp, N, MPI.DOUBLE],
                [b,      N, MPI.DOUBLE],
                op=MPI.SUM, root=0)

    if rank == 0:
        # Верификация последовательным вычислением
        b_seq = A.T @ x
        diff = np.max(np.abs(b - b_seq))
        print(f"compute time ~ {t1 - t0:.6f} s")
        print("max abs diff vs seq =", diff)

if __name__ == "__main__":
    main()
