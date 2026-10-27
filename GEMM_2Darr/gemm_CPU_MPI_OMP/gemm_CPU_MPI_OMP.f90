! Hybrid MPI + OpenMP matrix multiplication with OpenBLAS DGEMM.
! MPI distributes rows of A and C between ranks.
! OpenMP distributes columns of C between CPU threads.
! Each OpenMP thread calls single-threaded DGEMM.

program gemm_mpi_omp_blas
  use mpi
  use omp_lib
  implicit none

  integer, parameter :: dp = kind(1.0d0)

  integer :: n, rank, nprocs, ierr, provided
  integer :: base, rem, nloc, istart, lda
  integer :: i, j, k, nthr
  integer :: tid, nt, cols_base, cols_rem, ncols, jfirst

  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)

  real(dp) :: s1, s2, expected, err
  real(dp) :: local_err, global_err
  real(dp) :: t0, t1, t2
  real(dp) :: bcast_time, gemm_time, total_time
  real(dp) :: max_bcast_time, max_gemm_time, max_total_time
  real(dp) :: gflops_compute, gflops_total

  character(len=32) :: arg
  external :: dgemm

  ! Initialize MPI with main-thread-only MPI calls.
  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)

  if (provided < MPI_THREAD_FUNNELED) then
     print *, 'ERROR: MPI_THREAD_FUNNELED unavailable'
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! Global matrix dimension.
  n = 2048

  if (rank == 0 .and. command_argument_count() > 0) then
     call get_command_argument(1, arg)
     read(arg, *) n
  end if

  call MPI_Bcast(n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

  if (n < 1) then
     if (rank == 0) print *, 'ERROR: N must be positive'
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  ! Divide matrix rows among MPI ranks.
  base = n / nprocs
  rem  = mod(n, nprocs)

  if (rank < rem) then
     nloc   = base + 1
     istart = rank * (base + 1)
  else
     nloc   = base
     istart = rem * (base + 1) + (rank - rem) * base
  end if

  lda = max(1, nloc)

  allocate(a(lda,n), b(n,n), c(lda,n))

  ! Initialize local rows of A.
  !$omp parallel do private(i)
  do k = 1, n
     do i = 1, nloc
        a(i,k) = real(istart+i-1,dp) + real(k-1,dp)
     end do
  end do
  !$omp end parallel do

  ! Rank 0 initializes the global matrix B.
  if (rank == 0) then

     !$omp parallel do private(k)
     do j = 1, n
        do k = 1, n
           b(k,j) = real(k-1,dp) + 2.0_dp*real(j-1,dp)
        end do
     end do
     !$omp end parallel do

  end if

  c = 0.0_dp
  nthr = 0

  ! Synchronize MPI ranks before timing.
  call MPI_Barrier(MPI_COMM_WORLD, ierr)

  t0 = MPI_Wtime()

  ! Broadcast B from rank 0 to all MPI ranks.
  call MPI_Bcast(b, n*n, MPI_DOUBLE_PRECISION, &
                 0, MPI_COMM_WORLD, ierr)

  t1 = MPI_Wtime()

  ! OpenMP distributes local C columns among threads.
  ! Each thread performs its own single-threaded DGEMM.
  !$omp parallel private(tid,nt,cols_base,cols_rem,ncols,jfirst)

  tid = omp_get_thread_num()
  nt  = omp_get_num_threads()

  !$omp single
  nthr = nt
  !$omp end single

  cols_base = n / nt
  cols_rem  = mod(n, nt)

  ncols = cols_base

  if (tid < cols_rem) then
     ncols = ncols + 1
  end if

  jfirst = tid*cols_base + min(tid,cols_rem) + 1

  ! Each thread writes to different columns of C.
  if (nloc > 0 .and. ncols > 0) then

     call dgemm('N','N',nloc,ncols,n, &
                1.0_dp,a,lda, &
                b(1,jfirst),n, &
                0.0_dp,c(1,jfirst),lda)

  end if

  !$omp end parallel

  t2 = MPI_Wtime()

  bcast_time = t1 - t0
  gemm_time  = t2 - t1
  total_time = t2 - t0

  ! Analytical verification.
  s1 = real(n,dp)*real(n-1,dp)/2.0_dp

  s2 = real(n,dp)*real(n-1,dp) * &
       (2.0_dp*real(n,dp)-1.0_dp)/6.0_dp

  local_err = 0.0_dp

  do j = 1, n
     do i = 1, nloc

        expected = s2 + &
          (real(istart+i-1,dp) + 2.0_dp*real(j-1,dp))*s1 + &
          2.0_dp*real(n,dp)*real(istart+i-1,dp)*real(j-1,dp)

        err = abs(c(i,j)-expected) / &
              max(1.0_dp,abs(expected))

        local_err = max(local_err,err)

     end do
  end do

  ! Collect maximum error and execution times across ranks.
  call MPI_Reduce(local_err,global_err,1, &
                  MPI_DOUBLE_PRECISION,MPI_MAX, &
                  0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(bcast_time,max_bcast_time,1, &
                  MPI_DOUBLE_PRECISION,MPI_MAX, &
                  0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(gemm_time,max_gemm_time,1, &
                  MPI_DOUBLE_PRECISION,MPI_MAX, &
                  0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(total_time,max_total_time,1, &
                  MPI_DOUBLE_PRECISION,MPI_MAX, &
                  0,MPI_COMM_WORLD,ierr)

  ! Rank 0 reports global results.
  if (rank == 0) then

     gflops_compute = 0.0_dp
     gflops_total   = 0.0_dp

     if (max_gemm_time > 0.0_dp) then
        gflops_compute = 2.0_dp*real(n,dp)**3 / &
                         (max_gemm_time*1.0e9_dp)
     end if

     if (max_total_time > 0.0_dp) then
        gflops_total = 2.0_dp*real(n,dp)**3 / &
                       (max_total_time*1.0e9_dp)
     end if

     print '(A,I0)', 'Matrix dimension N = ', n
     print '(A,I0)', 'MPI ranks = ', nprocs
     print '(A,I0)', 'OpenMP threads per rank = ', nthr
     print '(A,I0)', 'Rows on rank 0 = ', nloc

     print '(A,ES18.8)', 'C(1,1) = ', c(1,1)
     print '(A,ES12.4)', 'Maximum relative error = ', global_err

     if (global_err < 1.0e-10_dp) then
        print *, 'Verification: PASS'
     else
        print *, 'Verification: FAIL'
     end if

     print '(A,F12.6,A)', &
       'MPI_Bcast time (max rank) = ', max_bcast_time, ' s'

     print '(A,F12.6,A)', &
       'DGEMM time (max rank) = ', max_gemm_time, ' s'

     print '(A,F12.6,A)', &
       'Total Bcast+DGEMM (max rank) = ', max_total_time, ' s'

     print '(A,F12.3)', &
       'DGEMM GFLOP/s (all ranks) = ', gflops_compute

     print '(A,F12.3)', &
       'Total GFLOP/s (all ranks) = ', gflops_total

  end if

  deallocate(a,b,c)

  call MPI_Finalize(ierr)

end program gemm_mpi_omp_blas
!1 node, 2 MPI ranks
!sbatch -N 1 -n 2 --ntasks-per-node=2 -c 2 run.slurm 2048

!2 nodes, 4 MPI ranks
!sbatch -N 2 -n 4 --ntasks-per-node=2 -c 2 run.slurm 2048

!4 nodes, 4 MPI ranks
!sbatch -N 4 -n 4 --ntasks-per-node=1 -c 2 run.slurm 2048