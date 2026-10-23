program dot_hybrid_gpu_dc
  use mpi
#ifdef USE_ACC
  use openacc
#endif
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  integer :: rank, nprocs, lrank, ierr, node_comm, i
  integer :: base, rem, nloc, istart, ndev, flen
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: acc, expected
  real(dp) :: lbuf(1), gbuf(1)
  character(len=16) :: mode
  character(len=8)  :: force
  logical :: aware

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  mode = 'staged'
  if (command_argument_count() > 0) call get_command_argument(1, mode)
  aware = (trim(mode) == 'aware')
  if (aware) then
     force = ' '
     call get_environment_variable('DOT_FORCE_AWARE', force, flen)
     if (trim(force) /= '1') then
        if (rank == 0) print *, 'Set DOT_FORCE_AWARE=1 to run aware mode.'
        call MPI_Finalize(ierr)
        stop
     end if
  end if
#ifndef USE_ACC
  if (aware) then
     if (rank == 0) print *, 'Aware mode needs build B (-DUSE_ACC with separate memory).'
     call MPI_Finalize(ierr)
     stop
  end if
#endif

  ! one GPU per rank, chosen before any data is allocated
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, &
                           MPI_INFO_NULL, node_comm, ierr)
  call MPI_Comm_rank(node_comm, lrank, ierr)
#ifdef USE_ACC
  ndev = acc_get_num_devices(acc_device_nvidia)
  if (ndev > 0) call acc_set_device_num(mod(lrank, ndev), acc_device_nvidia)
  print '(A,I0,A,I0,A,I0)', 'rank ', rank, ': GPUs visible = ', ndev, ', using device ', mod(lrank, max(ndev, 1))
#else
  ndev = 0
#endif

  ! MPI cut: block decomposition (same as dot_mpi.f90)
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
  lbuf(1) = 0.0_dp
  gbuf(1) = -1.0_dp
#ifdef USE_ACC
  !$acc enter data create(a, b, lbuf, gbuf)
#endif

  ! GPU cut: do concurrent, compiler maps iterations to GPU threads
  do concurrent (i = 1:nloc)
     a(i) = real(istart + i - 1, dp)
     b(i) = 1.0_dp
  end do

  acc = 0.0_dp
  do concurrent (i = 1:nloc) reduce(+:acc)
     acc = acc + a(i) * b(i)
  end do

  lbuf(1) = acc
  gbuf(1) = -1.0_dp

  if (.not. aware) then
     call MPI_Reduce(lbuf, gbuf, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                     MPI_COMM_WORLD, ierr)
  else
#ifdef USE_ACC
     !$acc update device(lbuf, gbuf)
     !$acc host_data use_device(lbuf, gbuf)
     call MPI_Reduce(lbuf, gbuf, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                     MPI_COMM_WORLD, ierr)
     !$acc end host_data
     if (rank == 0) print '(A,F18.1)', 'CPU copy right after reduce = ', gbuf(1)
     !$acc update host(gbuf)
#endif
  end if

  if (rank == 0) then
     expected = real(n, dp) * real(n - 1, dp) / 2.0_dp
     print '(A,A,A,I0)', 'mode = ', trim(mode), ', ranks = ', nprocs
     print '(A,F18.1)', 'Dot product = ', gbuf(1)
     print '(A,F18.1)', 'Expected    = ', expected
     if (gbuf(1) == expected) then
        print *, 'Verification: PASS'
     else
        print *, 'Verification: FAIL'
     end if
  end if

#ifdef USE_ACC
  !$acc exit data delete(a, b, lbuf, gbuf)
#endif
  deallocate(a, b)
  call MPI_Finalize(ierr)
end program dot_hybrid_gpu_dc