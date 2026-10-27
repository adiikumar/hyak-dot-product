program gemm_mpi_loop
  use mpi
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: n, rank, nprocs, ierr, ios
  integer :: base, rem, nloc, istart, lda, i, j, k
  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)
  real(dp) :: s, s1, s2, expected, error, local_error, global_error
  real(dp) :: t0, t1, t2, times(3), max_times(3)
  character(len=32) :: arg

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nprocs,ierr)

  n = 2048
  if (rank == 0 .and. command_argument_count() >= 1) then
     call get_command_argument(1,arg)
     read(arg,*,iostat=ios) n
     if (ios /= 0) n = 0
  end if

  call MPI_Bcast(n,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

  if (n < nprocs) then
     if (rank == 0) print *, 'ERROR: N must be at least the number of ranks'
     call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end if

  ! MPI row decomposition.
  base = n/nprocs
  rem = mod(n,nprocs)

  nloc = base
  if (rank < rem) nloc = nloc+1

  istart = rank*base + min(rank,rem)
  lda = max(1,nloc)

  allocate(a(lda,n),b(n,n),c(lda,n))

  ! Each rank initializes its own rows of A.
  do k = 1, n
     do i = 1, nloc
        a(i,k) = real(istart+i-1,dp) + real(k-1,dp)
     end do
  end do

  ! Rank 0 initializes B.
  if (rank == 0) then
     do j = 1, n
        do k = 1, n
           b(k,j) = real(k-1,dp) + 2.0_dp*real(j-1,dp)
        end do
     end do
  end if

  call MPI_Barrier(MPI_COMM_WORLD,ierr)

  t0 = MPI_Wtime()
  call MPI_Bcast(b,n*n,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
  t1 = MPI_Wtime()

  ! Explicit local matrix multiplication.
  do j = 1, n
     do i = 1, nloc
        s = 0.0_dp
        do k = 1, n
           s = s + a(i,k)*b(k,j)
        end do
        c(i,j) = s
     end do
  end do

  t2 = MPI_Wtime()

  times = [t1-t0,t2-t1,t2-t0]

  call MPI_Reduce(times,max_times,3,MPI_DOUBLE_PRECISION, &
                  MPI_MAX,0,MPI_COMM_WORLD,ierr)

  ! Analytical verification.
  s1 = real(n,dp)*real(n-1,dp)/2.0_dp
  s2 = real(n,dp)*real(n-1,dp)*(2.0_dp*real(n,dp)-1.0_dp)/6.0_dp

  local_error = 0.0_dp

  do j = 1, n
     do i = 1, nloc
        expected = s2 + &
          (real(istart+i-1,dp)+2.0_dp*real(j-1,dp))*s1 &
          + 2.0_dp*real(n,dp)*real(istart+i-1,dp)*real(j-1,dp)

        error = abs(c(i,j)-expected)/max(1.0_dp,abs(expected))
        local_error = max(local_error,error)
     end do
  end do

  call MPI_Reduce(local_error,global_error,1,MPI_DOUBLE_PRECISION, &
                  MPI_MAX,0,MPI_COMM_WORLD,ierr)

  if (rank == 0) then
     print '(A,I0)', 'Matrix dimension N = ', n
     print '(A,I0)', 'MPI ranks = ', nprocs
     print '(A,I0)', 'Rows on rank 0 = ', nloc
     print '(A,ES18.8)', 'C(1,1) = ', c(1,1)
     print '(A,ES12.4)', 'Maximum relative error = ', global_error

     if (global_error < 1.0e-10_dp) then
        print *, 'Verification: PASS'
     else
        print *, 'Verification: FAIL'
        call MPI_Abort(MPI_COMM_WORLD,2,ierr)
     end if

     print '(A,F12.6,A)', 'MPI_Bcast time (max rank) = ', max_times(1), ' s'
     print '(A,F12.6,A)', 'Loop GEMM time (max rank) = ', max_times(2), ' s'
     print '(A,F12.6,A)', 'Total Bcast+GEMM (max rank) = ', max_times(3), ' s'

     print '(A,F12.3)', 'Loop GEMM GFLOP/s (all ranks) = ', &
          2.0_dp*real(n,dp)**3/(max_times(2)*1.0e9_dp)

     print '(A,F12.3)', 'Total GFLOP/s (all ranks) = ', &
          2.0_dp*real(n,dp)**3/(max_times(3)*1.0e9_dp)
  end if

  deallocate(a,b,c)
  call MPI_Finalize(ierr)

end program gemm_mpi_loop
