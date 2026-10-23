! dot_hybrid.f90 : hybrid MPI + OpenMP dot product.
! MPI splits the index range across ranks (distributed memory).
! OpenMP splits each rank's local loop across threads (shared memory).
! Thread count comes from the OMP_NUM_THREADS environment variable.
program dot_hybrid
  use mpi
  use omp_lib
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  integer :: rank, nprocs, ierr, provided
  integer :: base, rem, nloc, istart, i, nthr
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: local, global, expected, t0, t1

  ! FUNNELED: only the main thread makes MPI calls, outside parallel regions
  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! Block decomposition across ranks: first mod(n,nprocs) ranks get one extra
  base = n / nprocs
  rem  = mod(n, nprocs)
  if (rank < rem) then
     nloc   = base + 1
     istart = rank * (base + 1)
  else
     nloc   = base
     istart = rem * (base + 1) + (rank - rem) * base
  end if

  allocate(a(nloc), b(nloc))

  ! Same data as serial: a = global index (0-based), b = 1
  !$omp parallel do
  do i = 1, nloc
     a(i) = real(istart + i - 1, dp)
     b(i) = 1.0_dp
  end do
  !$omp end parallel do

  ! Local dot product, threaded with an OpenMP reduction
  local = 0.0_dp
  nthr  = 0
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t0 = MPI_Wtime()
  !$omp parallel
  !$omp single
  nthr = omp_get_num_threads()
  !$omp end single
  !$omp do reduction(+:local)
  do i = 1, nloc
     local = local + a(i) * b(i)
  end do
  !$omp end do
  !$omp end parallel

  ! Combine the per-rank partial sums on rank 0
  call MPI_Reduce(local, global, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                  MPI_COMM_WORLD, ierr)
  t1 = MPI_Wtime()

  if (rank == 0) then
     expected = real(n, dp) * real(n - 1, dp) / 2.0_dp
     print '(A,I0,A,I0)', 'MPI ranks = ', nprocs, ', OpenMP threads per rank = ', nthr
     print '(A,F18.1)', 'Hybrid dot product = ', global
     print '(A,F18.1)', 'Expected           = ', expected
     if (global == expected) then
        print '(A,F10.6,A)', 'Verification: PASS   (time ', t1 - t0, ' s)'
     else
        print '(A,F10.6,A)', 'Verification: FAIL   (time ', t1 - t0, ' s)'
     end if
  end if

  deallocate(a, b)
  call MPI_Finalize(ierr)
end program dot_hybrid
