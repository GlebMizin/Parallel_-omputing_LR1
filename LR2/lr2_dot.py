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
    # === ВАЖНО: размер вектора (М) ===
    # По лекции можно взять M=300000, но на твоей машине можно меньше/больше.
    M = 300000

    # root создаёт вектор a
    if rank == 0:
        a = np.arange(1, M + 1, dtype=np.float64)
        counts, displs = counts_displs(M, size)
    else:
        a = None
        counts = None
        displs = None

    # всем нужны counts/displs
    counts = comm.bcast(counts, root=0)
    displs = comm.bcast(displs, root=0)

    m_part = int(counts[rank])
    a_part = np.empty(m_part, dtype=np.float64)

    # Scatterv кусочков a
    comm.Scatterv([a, counts, displs, MPI.DOUBLE],
                  [a_part, m_part, MPI.DOUBLE],
                  root=0)

    # локальная часть скалярного произведения
    local = np.dot(a_part, a_part)

    # собрать сумму на root
    total = comm.reduce(local, op=MPI.SUM, root=0)

    if rank == 0:
        print("dot(a,a) =", total)

if __name__ == "__main__":
    main()
