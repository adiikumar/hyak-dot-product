! MPI distributes matrix rows.
! OpenMP Target executes GEMM on one or more GPUs per rank.
program gemm_mpi_gpu_omp
  use mpi
  use omp_lib
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: rank, nprocs, ierr, provided
  integer :: node_comm, lrank, ranks_node
  integer :: n, ngpr, ndev, first_dev, dev
  integer :: g, i, j, k, nloc, istart, lda
  integer :: base, rem, jbase, jrem, j0, j1, ios
  integer, allocatable :: dev_ids(:), col_first(:), col_last(:)
  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)
  real(dp) :: s, s1, s2, expected, err
  real(dp) :: local_err, global_err
  real(dp) :: t0, t1, t2, t3, t4, t5
  real(dp) :: local_times(5), max_times(5), gflops
  logical :: onhost
  character(len=32) :: arg
  character(len=128) :: visible

  ! Only the main CPU thread makes MPI calls.
  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
  if (provided < MPI_THREAD_FUNNELED) then
     print *, 'MPI_THREAD_FUNNELED unavailable'
     call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! Identify MPI ranks residing on the same node.
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, &
                           rank, MPI_INFO_NULL, node_comm, ierr)
  call MPI_Comm_rank(node_comm, lrank, ierr)
  call MPI_Comm_size(node_comm, ranks_node, ierr)

  n = 2048
  if (rank == 0 .and. command_argument_count() >= 1) then
     call get_command_argument(1, arg)
     read(arg, *, iostat=ios) n
     if (ios /= 0) n = 0
  end if
  call MPI_Bcast(n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

  ! Read the number of GPUs assigned to each MPI rank.
  ngpr = 1
  call get_environment_variable('GPUS_PER_RANK', arg, status=ios)
  if (ios == 0) then
     read(arg, *, iostat=ios) ngpr
     if (ios /= 0) ngpr = 0
  end if

  if (n < nprocs .or. ngpr < 1 .or. ngpr > n) then
     if (rank == 0) print *, 'ERROR: invalid N or GPUs/rank'
     call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if

  ! Select GPUs using the rank's position within its node.
  ndev = omp_get_num_devices()

  if (ndev >= ranks_node*ngpr) then
     first_dev = lrank*ngpr
  else if (ndev == ngpr) then
     ! Device visibility already restricted per rank.
     first_dev = 0
  else
     print *, 'rank', rank, ': insufficient visible GPUs:', ndev
     call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
  end if

  allocate(dev_ids(ngpr), col_first(ngpr), col_last(ngpr))

  ! Confirm each selected device executes OpenMP Target.
  do g = 1, ngpr
     dev_ids(g) = first_dev + g - 1
     onhost = .true.

     !$omp target map(from:onhost) device(dev_ids(g))
     onhost = omp_is_initial_device()
     !$omp end target

     if (onhost) then
        print *, 'rank', rank, ': GPU offload failed'
        call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
     end if
  end do

  visible = ''
  call get_environment_variable('CUDA_VISIBLE_DEVICES', visible)
  print '(A,I0,A,I0,A,I0,A,A)', 'rank=', rank, &
        ' local_rank=', lrank, ' GPUs=', ngpr, &
        ' visible=', trim(visible)
  print '(A,I0,A,*(I0,1X))', 'rank=', rank, &
        ' OpenMP device IDs: ', dev_ids

  ! Same row decomposition as the MPI dot-product program.
  base = n / nprocs
  rem = mod(n, nprocs)

  nloc = base
  if (rank < rem) nloc = nloc + 1
  istart = rank*base + min(rank,rem)

  lda = max(1,nloc)
  allocate(a(lda,n), b(n,n), c(lda,n))

  ! Initialize the rank-local rows of A.
  do k = 1, n
     do i = 1, nloc
        a(i,k) = real(istart+i-1,dp) + real(k-1,dp)
     end do
  end do

  ! Rank 0 initializes matrix B.
  if (rank == 0) then
     do j = 1, n
        do k = 1, n
           b(k,j) = real(k-1,dp) + 2.0_dp*real(j-1,dp)
        end do
     end do
  end if
  c = 0.0_dp

  ! Partition local output columns across this rank's GPUs.
  jbase = n/ngpr
  jrem = mod(n,ngpr)

  do g = 1, ngpr
     col_first(g) = (g-1)*jbase + min(g-1,jrem) + 1
     col_last(g) = col_first(g) + jbase - 1
     if (g <= jrem) col_last(g) = col_last(g) + 1
  end do

  ! MPI communication: distribute B between CPU processes.
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t0 = MPI_Wtime()
  call MPI_Bcast(b, n*n, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
  t1 = MPI_Wtime()

  ! Copy A and B onto each assigned GPU.
  !$omp parallel do num_threads(ngpr) schedule(static) private(dev)
  do g = 1, ngpr
     dev = dev_ids(g)
     !$omp target enter data map(to:a,b) map(alloc:c) device(dev)
  end do
  !$omp end parallel do
  t2 = MPI_Wtime()

  ! GPU-only GEMM arithmetic using OpenMP Target threads.
  !$omp parallel do num_threads(ngpr) schedule(static) private(dev,j0,j1)
  do g = 1, ngpr
     dev = dev_ids(g)
     j0 = col_first(g)
     j1 = col_last(g)

     !$omp target teams distribute parallel do collapse(2) &
     !$omp& device(dev) private(k,s) map(alloc:a,b,c)
     do j = j0, j1
        do i = 1, nloc
           s = 0.0_dp
           do k = 1, n
              s = s + a(i,k)*b(k,j)
           end do
           c(i,j) = s
        end do
     end do
     !$omp end target teams distribute parallel do
  end do
  !$omp end parallel do
  t3 = MPI_Wtime()

  ! Download each GPU's assigned columns of C.
  !$omp parallel do num_threads(ngpr) schedule(static) private(dev,j0,j1)
  do g = 1, ngpr
     dev = dev_ids(g)
     j0 = col_first(g)
     j1 = col_last(g)

     !$omp target update from(c(1:nloc,j0:j1)) device(dev)
  end do
  !$omp end parallel do
  t4 = MPI_Wtime()

  ! Release GPU allocations.
  !$omp parallel do num_threads(ngpr) schedule(static) private(dev)
  do g = 1, ngpr
     dev = dev_ids(g)
     !$omp target exit data map(delete:a,b,c) device(dev)
  end do
  !$omp end parallel do
  t5 = MPI_Wtime()

  ! Analytical verification, identical to CPU GEMM.
  s1 = real(n,dp)*real(n-1,dp)/2.0_dp
  s2 = real(n,dp)*real(n-1,dp) * &
       (2.0_dp*real(n,dp)-1.0_dp)/6.0_dp

  local_err = 0.0_dp
  do j = 1, n
     do i = 1, nloc
        expected = s2 + &
           (real(istart+i-1,dp)+2.0_dp*real(j-1,dp))*s1 + &
           2.0_dp*real(n,dp)*real(istart+i-1,dp)*real(j-1,dp)

        err = abs(c(i,j)-expected) / &
              max(1.0_dp,abs(expected))
        local_err = max(local_err,err)
     end do
  end do

  local_times = [t1-t0, t2-t1, t3-t2, t4-t3, t5-t0]

  call MPI_Reduce(local_times,max_times,5,MPI_DOUBLE_PRECISION, &
                  MPI_MAX,0,MPI_COMM_WORLD,ierr)
  call MPI_Reduce(local_err,global_err,1,MPI_DOUBLE_PRECISION, &
                  MPI_MAX,0,MPI_COMM_WORLD,ierr)

  if (rank == 0) then
     print '(A,I0,A,I0,A,I0)', 'N=', n, &
           ' MPI ranks=', nprocs, ' GPUs/rank=', ngpr
     print '(A,ES12.4)', 'Maximum relative error = ', global_err

     if (global_err < 1.0e-10_dp) then
        print *, 'Verification: PASS'
     else
        print *, 'Verification: FAIL'
     end if

     print '(A,F12.6,A)', 'MPI_Bcast (max rank) = ', max_times(1), ' s'
     print '(A,F12.6,A)', 'Host-to-GPU (max rank) = ', max_times(2), ' s'
     print '(A,F12.6,A)', 'GPU GEMM (max rank) = ', max_times(3), ' s'
     print '(A,F12.6,A)', 'GPU-to-host (max rank) = ', max_times(4), ' s'
     print '(A,F12.6,A)', 'Total incl. transfers (max rank) = ', max_times(5), ' s'

     if (max_times(3) > 0.0_dp) then
        gflops = 2.0_dp*real(n,dp)**3 / &
                 (max_times(3)*1.0e9_dp)
        print '(A,F12.3)', 'GPU GEMM GFLOP/s = ', gflops
     end if
  end if

  deallocate(a,b,c,dev_ids,col_first,col_last)
  call MPI_Comm_free(node_comm, ierr)
  call MPI_Finalize(ierr)

end program gemm_mpi_gpu_omp
! Two GPUs on one node
!sbatch -N 1 --ntasks-per-node=2 --gpus-per-node=2 run.slurm 2048 1
!Two GPUs on two nodes
!  sbatch -N 2 --ntasks-per-node=1 --gpus-per-node=1 run.slurm 2048 1
!two GPUs per MPI rank
!  sbatch --gpus-per-node=4 --cpus-per-task=2 run.slurm 2048 2
!sbatch -N 1 -n 2 --ntasks-per-node=2 --gpus-per-node=2 -c 1 run.slurm 2048 1
!sbatch -N 2 -n 4 --ntasks-per-node=2 --gpus-per-node=2 -c 1 run.slurm 2048 1
!sbatch -N 4 -n 4 --ntasks-per-node=1 --gpus-per-node=1 -c 1 run.slurm 2048 1