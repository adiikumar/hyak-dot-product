! dot_hybrid_gpu_cap.f90 : MPI + OpenMP GPU offload dot product with optional caps on GPU teams and threads.
! DOT_TEAMS and DOT_THREADS (environment variables) set num_teams and thread_limit. Unset = no cap.
program dot_hybrid_gpu_cap
  use omp_lib
  use mpi
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  integer :: rank, nprocs, lrank, ndev, ierr, node_comm, i
  integer :: base, rem, nloc, istart, flen
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: acc, expected
  real(dp) :: lbuf(1), gbuf(1)
  character(len=16) :: mode
  character(len=8)  :: force
  logical :: aware, onhost
  integer :: nteams, nthr, nteams_seen, nthr_seen, elen
  character(len=16) :: ebuf

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

  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, &
                           MPI_INFO_NULL, node_comm, ierr)
  call MPI_Comm_rank(node_comm, lrank, ierr)
  ndev = omp_get_num_devices()
  if (ndev > 0) call omp_set_default_device(mod(lrank, ndev))

  onhost = .true.
  !$omp target map(from: onhost)
  onhost = omp_is_initial_device()
  !$omp end target
  if (onhost) then
     print '(A,I0,A)', 'rank ', rank, ': target region ran on the HOST (no GPU used)'
  else
     print '(A,I0,A)', 'rank ', rank, ': target region ran on the GPU'
  end if

  nteams = 0
  nthr   = 0
  ebuf = ' '
  call get_environment_variable('DOT_TEAMS', ebuf, elen)
  if (elen > 0) read(ebuf, *) nteams
  ebuf = ' '
  call get_environment_variable('DOT_THREADS', ebuf, elen)
  if (elen > 0) read(ebuf, *) nthr

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
  acc = 0.0d0
  lbuf(1) = 0.0d0
  gbuf(1) = -1.0d0
  !$omp target enter data map(alloc: a, b, lbuf, gbuf) map(to: acc)

  if (nteams > 0 .and. nthr > 0) then
     !$omp target teams distribute parallel do num_teams(nteams) thread_limit(nthr)
     do i = 1, nloc
        a(i) = real(istart + i - 1, dp)
        b(i) = 1.0d0
     end do
     !$omp end target teams distribute parallel do

     !$omp target teams distribute parallel do reduction(+:acc) num_teams(nteams) thread_limit(nthr)
     do i = 1, nloc
        acc = acc + a(i) * b(i)
     end do
     !$omp end target teams distribute parallel do
  else
     !$omp target teams distribute parallel do
     do i = 1, nloc
        a(i) = real(istart + i - 1, dp)
        b(i) = 1.0d0
     end do
     !$omp end target teams distribute parallel do

     !$omp target teams distribute parallel do reduction(+:acc)
     do i = 1, nloc
        acc = acc + a(i) * b(i)
     end do
     !$omp end target teams distribute parallel do
  end if

  nteams_seen = 0
  nthr_seen = 0
  if (nteams > 0 .and. nthr > 0) then
     !$omp target teams num_teams(nteams) thread_limit(nthr) map(from: nteams_seen, nthr_seen)
     if (omp_get_team_num() == 0) then
        nteams_seen = omp_get_num_teams()
        !$omp parallel
        !$omp single
        nthr_seen = omp_get_num_threads()
        !$omp end single
        !$omp end parallel
     end if
     !$omp end target teams
  else
     !$omp target teams map(from: nteams_seen, nthr_seen)
     if (omp_get_team_num() == 0) then
        nteams_seen = omp_get_num_teams()
        !$omp parallel
        !$omp single
        nthr_seen = omp_get_num_threads()
        !$omp end single
        !$omp end parallel
     end if
     !$omp end target teams
  end if
  print '(A,I0,A,I0,A,I0,A,I0)', 'rank ', rank, ': teams = ', nteams_seen, &
        ', threads per team = ', nthr_seen, ', GPU threads = ', nteams_seen * nthr_seen

  !$omp target map(tofrom: acc)
  lbuf(1) = acc
  gbuf(1) = -1.0d0
  !$omp end target

  if (.not. aware) then
     !$omp target update from(lbuf)
     call MPI_Reduce(lbuf, gbuf, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                     MPI_COMM_WORLD, ierr)
  else
     !$omp target data use_device_ptr(lbuf, gbuf)
     call MPI_Reduce(lbuf, gbuf, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                     MPI_COMM_WORLD, ierr)
     !$omp end target data
     if (rank == 0) print '(A,F18.1)', 'CPU copy right after reduce = ', gbuf(1)
     !$omp target update from(gbuf)
  end if

  if (rank == 0) then
     expected = real(n, dp) * real(n - 1, dp) / 2.0d0
     print '(A,A,A,I0)', 'mode = ', trim(mode), ', ranks = ', nprocs
     print '(A,F18.1)', 'Dot product = ', gbuf(1)
     print '(A,F18.1)', 'Expected    = ', expected
     if (gbuf(1) == expected) then
        print *, 'Verification: PASS'
     else
        print *, 'Verification: FAIL'
     end if
  end if

  !$omp target exit data map(delete: a, b, lbuf, gbuf, acc)
  deallocate(a, b)
  call MPI_Finalize(ierr)
end program dot_hybrid_gpu_cap
