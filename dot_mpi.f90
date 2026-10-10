! dot_mpi.f90 : parallel dot product with MPI (block decomposition + MPI_Reduce)
program dot_mpi
  use mpi
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  integer :: rank, nprocs, ierr
  integer :: base, rem, nloc, istart, i
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: local, global, expected

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! Block decomposition: first mod(n,nprocs) ranks get one extra element
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
  do i = 1, nloc
     a(i) = real(istart + i - 1, dp)
     b(i) = 1.0_dp
  end do

  local = 0.0_dp
  do i = 1, nloc
     local = local + a(i) * b(i)
  end do

  call MPI_Reduce(local, global, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                  MPI_COMM_WORLD, ierr)

  if (rank == 0) then
     expected = real(n, dp) * real(n - 1, dp) / 2.0_dp
     print '(A,I0)',    'MPI ranks = ', nprocs
     print '(A,F18.1)', 'MPI dot product = ', global
     print '(A,F18.1)', 'Expected        = ', expected
     if (global == expected) then
        print *, 'Verification: PASS'
     else
        print *, 'Verification: FAIL'
     end if
  end if

  deallocate(a, b)
  call MPI_Finalize(ierr)
end program dot_mpi
