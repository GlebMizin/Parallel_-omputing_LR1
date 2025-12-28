from mpi4py import MPI
import numpy as np
import os
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def counts_displs(n: int, p: int):
    base, rem = divmod(n, p)
    counts = np.array([base + 1 if r < rem else base for r in range(p)], dtype=np.int32)
    displs = np.zeros(p, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

def generate_files(N: int, M: int, seed: int = 123):
    """
    Генерим A (M×N), x_true (N), b = A@x_true + шум.
    Пишем:
      in.dat   : N, M
      AData.dat: M*N чисел
      bData.dat: M чисел
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, N), dtype=np.float64)
    x_true = rng.standard_normal(N, dtype=np.float64)
    b = A @ x_true + 0.01 * rng.standard_normal(M, dtype=np.float64)

    with open("in.dat", "w", encoding="utf-8") as f:
        f.write(f"{N}\n{M}\n")

    # по одному числу в строке (как в лекции/лабах)
    np.savetxt("AData.dat", A.reshape(-1), fmt="%.18e")
    np.savetxt("bData.dat", b, fmt="%.18e")

def read_files():
    with open("in.dat", "r", encoding="utf-8") as f:
        N = int(f.readline().strip())
        M = int(f.readline().strip())
    A_flat = np.loadtxt("AData.dat", dtype=np.float64)
    b = np.loadtxt("bData.dat", dtype=np.float64)
    A = A_flat.reshape(M, N)
    return N, M, A, b

def cg_least_squares_parallel(A_part, b_part, N, N_part, rcounts_N, displs_N,
                             max_iter=500, tol=1e-8):
    """
    Решаем МНК: min ||Ax - b||_2 через нормальные уравнения:
        (A^T A) x = A^T b
    CG работает на операторе B = A^T A.
    Данные:
      A_part: (M_part×N), b_part: (M_part)
      x хранится по частям (x_part), но для matvec собираем полный вектор Allgatherv.
    """
    # локальные размеры
    M_part = b_part.size

    # x хранится по частям
    x_part = np.zeros(N_part, dtype=np.float64)
    x = np.empty(N, dtype=np.float64)

    # c = A^T b
    c_temp = A_part.T @ b_part                  # (N,)
    c = np.empty(N, dtype=np.float64)
    comm.Allreduce([c_temp, MPI.DOUBLE], [c, MPI.DOUBLE], op=MPI.SUM)

    # r0 = c - Bx0, x0=0 => r0=c
    r_part = c[displs_N[rank]:displs_N[rank] + N_part].copy()
    p_part = r_part.copy()

    rr = comm.allreduce(float(np.dot(r_part, r_part)), op=MPI.SUM)

    iters = 0
    start = MPI.Wtime()

    for k in range(max_iter):
        iters = k + 1

        # p_full на всех процессах
        comm.Allgatherv([p_part, MPI.DOUBLE],
                        [x, rcounts_N, displs_N, MPI.DOUBLE])  # используем x как буфер под p_full
        p_full = x

        # y_part = A_part @ p_full (M_part)
        y_part = A_part @ p_full

        # Bp = A^T y
        Bp_temp = A_part.T @ y_part             # (N,)
        Bp_full = np.empty(N, dtype=np.float64)
        comm.Allreduce([Bp_temp, MPI.DOUBLE], [Bp_full, MPI.DOUBLE], op=MPI.SUM)
        Bp_part = Bp_full[displs_N[rank]:displs_N[rank] + N_part]

        pBp = comm.allreduce(float(np.dot(p_part, Bp_part)), op=MPI.SUM)

        # защита от деления на 0
        if pBp == 0.0:
            break

        alpha = rr / pBp

        # x = x + alpha p
        x_part += alpha * p_part

        # r = r - alpha Bp
        r_part -= alpha * Bp_part

        rr_new = comm.allreduce(float(np.dot(r_part, r_part)), op=MPI.SUM)

        if np.sqrt(rr_new) < tol:
            rr = rr_new
            break

        beta = rr_new / rr
        p_part = r_part + beta * p_part
        rr = rr_new

    comm.Barrier()
    end = MPI.Wtime()
    return x_part, iters, (end - start), np.sqrt(rr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=200, help="число неизвестных (N)")
    parser.add_argument("--M", type=int, default=300, help="число уравнений (M)")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--regen", action="store_true", help="перегенерить файлы заново")
    args = parser.parse_args()

    # === root: гарантируем наличие файлов ===
    if rank == 0:
        need = args.regen or (not os.path.exists("in.dat")) or (not os.path.exists("AData.dat")) or (not os.path.exists("bData.dat"))
        if need:
            generate_files(N=args.N, M=args.M, seed=123)

    comm.Barrier()

    # === читаем N/M на root, рассылаем всем ===
    if rank == 0:
        N, M, A, b = read_files()
    else:
        N = None
        M = None
        A = None
        b = None

    N = comm.bcast(N, root=0)
    M = comm.bcast(M, root=0)

    # разбиение по строкам матрицы (M) и по координатам x (N)
    rcounts_M, displs_M = counts_displs(M, size)
    rcounts_N, displs_N = counts_displs(N, size)
    M_part = int(rcounts_M[rank])
    N_part = int(rcounts_N[rank])

    # === Scatterv A по строкам ===
    A_part = np.empty((M_part, N), dtype=np.float64)
    sendcounts_A = (rcounts_M * N).astype(np.int32)
    displs_A = (displs_M * N).astype(np.int32)

    if rank == 0:
        comm.Scatterv([A.reshape(-1), sendcounts_A, displs_A, MPI.DOUBLE],
                      [A_part.reshape(-1), M_part * N, MPI.DOUBLE], root=0)
    else:
        comm.Scatterv([None, None, None, None],
                      [A_part.reshape(-1), M_part * N, MPI.DOUBLE], root=0)

    # === Scatterv b ===
    b_part = np.empty(M_part, dtype=np.float64)
    if rank == 0:
        comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
                      [b_part, M_part, MPI.DOUBLE], root=0)
    else:
        comm.Scatterv([None, None, None, None],
                      [b_part, M_part, MPI.DOUBLE], root=0)

    # === CG ===
    x_part, iters, t_core, rnorm = cg_least_squares_parallel(
        A_part, b_part,
        N=N, N_part=N_part,
        rcounts_N=rcounts_N, displs_N=displs_N,
        max_iter=args.max_iter, tol=args.tol
    )

    # === Собираем x на root ===
    if rank == 0:
        x_cg = np.empty(N, dtype=np.float64)
    else:
        x_cg = None

    comm.Gatherv([x_part, N_part, MPI.DOUBLE],
                 [x_cg, rcounts_N, displs_N, MPI.DOUBLE], root=0)

    if rank == 0:
        # сравнение с lstsq
        x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        diff = np.max(np.abs(x_cg - x_np))

        res_cg = np.linalg.norm(A @ x_cg - b)
        res_np = np.linalg.norm(A @ x_np - b)

        print(f"N={N}, M={M}, procs={size}")
        print(f"CG iters={iters}, core_time={t_core:.6f} s, ||r||={rnorm:.3e}")
        print(f"max|x_cg - x_lstsq| = {diff:.3e}")
        print(f"||Ax_cg-b|| = {res_cg:.3e}, ||Ax_lstsq-b|| = {res_np:.3e}")

if __name__ == "__main__":
    main()
