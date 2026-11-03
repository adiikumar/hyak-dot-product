program gradient_CPU_MPI_OMP_GPU
  use iso_fortran_env, only: real64
  use mpi
  use omp_lib
  implicit none

  type gpu_slab
    real(real64), allocatable :: p(:,:,:)
    real(real64), allocatable :: gx(:,:,:), gy(:,:,:), gz(:,:,:)
    real(real64), allocatable :: sendlo(:,:), sendhi(:,:)
    real(real64), allocatable :: recvlo(:,:), recvhi(:,:)
    integer :: ny, offset, dev
  end type

  type(gpu_slab), allocatable :: slab(:)

  integer :: ierr, rank, nranks, provided, n, ng, ndev
  integer :: ios, d, q, rem, ny_rank, offset_rank
  integer :: prev, next, stat, status(MPI_STATUS_SIZE)
  character(64) :: arg

  real(real64) :: h, twopi
  real(real64) :: linf, l1, l2, exact2
  real(real64) :: di, d1, d2, de
  real(real64) :: rel2, expected
  real(real64) :: start_time, numerical_start
  real(real64) :: init_time, pack_time, comm_time
  real(real64) :: unpack_time, fdm_time, norm_time
  real(real64) :: numerical_time, application_time
  real(real64) :: t0
  real(real64) :: local_times(8), global_times(8)

  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
  if (provided < MPI_THREAD_FUNNELED) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  n = 1024
  ng = 1

  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg)
    read(arg, *, iostat=ios) n
    if (ios /= 0) n = -1
  end if

  if (command_argument_count() >= 2) then
    call get_command_argument(2, arg)
    read(arg, *, iostat=ios) ng
    if (ios /= 0) ng = -1
  end if

  if (n < 4 .or. ng < 1 .or. nranks*ng > n) then
    if (rank == 0) print *, 'ERROR: Invalid mesh or GPU count'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if

  ndev = omp_get_num_devices()

  if (ndev < ng) then
    print *, 'Rank=', rank, ' visible GPUs=', ndev, &
             ' requested GPUs=', ng
    call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
  end if

  h = 1.0_real64 / real(n, real64)
  twopi = 2.0_real64 * acos(-1.0_real64)

  ! MPI decomposition along y.
  q = n/nranks
  rem = mod(n,nranks)

  ny_rank = q
  if (rank < rem) ny_rank = ny_rank+1

  offset_rank = rank*q + min(rank,rem)

  prev = modulo(rank-1,nranks)
  next = modulo(rank+1,nranks)

  allocate(slab(ng))

  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  start_time = MPI_Wtime()

  ! Subdivide each MPI rank's y-slab among its GPUs.
  do d = 1,ng

    slab(d)%ny = ny_rank/ng
    if (d <= mod(ny_rank,ng)) slab(d)%ny = slab(d)%ny+1

    slab(d)%offset = offset_rank + (d-1)*(ny_rank/ng) &
                   + min(d-1,mod(ny_rank,ng))

    slab(d)%dev = d-1

    allocate( &
      slab(d)%p(n+2,slab(d)%ny+2,n+2), &
      slab(d)%gx(n,slab(d)%ny,n), &
      slab(d)%gy(n,slab(d)%ny,n), &
      slab(d)%gz(n,slab(d)%ny,n), &
      slab(d)%sendlo(n,n), slab(d)%sendhi(n,n), &
      slab(d)%recvlo(n,n), slab(d)%recvhi(n,n), &
      stat=stat)

    if (stat /= 0) then
      print *, 'Allocation failed: rank=',rank,' device=',d-1
      call MPI_Abort(MPI_COMM_WORLD,4,ierr)
    end if

    call enter_device( &
      slab(d)%p,slab(d)%gx,slab(d)%gy,slab(d)%gz, &
      slab(d)%sendlo,slab(d)%sendhi, &
      slab(d)%recvlo,slab(d)%recvhi,slab(d)%dev)
  end do

  t0 = MPI_Wtime()

  do d = 1,ng
    call initialize_pressure( &
      slab(d)%p,slab(d)%ny,slab(d)%offset,slab(d)%dev)
  end do

  init_time = MPI_Wtime()-t0

  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  numerical_start = MPI_Wtime()

  ! Pack y-faces on each GPU and transfer them to host.
  t0 = MPI_Wtime()

  do d = 1,ng
    call pack_faces( &
      slab(d)%p,slab(d)%sendlo,slab(d)%sendhi, &
      slab(d)%ny,slab(d)%dev)
  end do

  pack_time = MPI_Wtime()-t0

  ! Communication between GPUs inside the same MPI rank.
  ! Host buffers are used; no GPU-aware MPI.
  t0 = MPI_Wtime()

  do d = 1,ng-1
    slab(d)%recvhi = slab(d+1)%sendlo
    slab(d+1)%recvlo = slab(d)%sendhi
  end do

  ! Communication between MPI ranks.
  ! Also used for periodic self-exchange when nranks=1.
  call MPI_Sendrecv( &
    slab(ng)%sendhi,n*n,MPI_DOUBLE_PRECISION,next,10, &
    slab(1)%recvlo,n*n,MPI_DOUBLE_PRECISION,prev,10, &
    MPI_COMM_WORLD,status,ierr)

  call MPI_Sendrecv( &
    slab(1)%sendlo,n*n,MPI_DOUBLE_PRECISION,prev,11, &
    slab(ng)%recvhi,n*n,MPI_DOUBLE_PRECISION,next,11, &
    MPI_COMM_WORLD,status,ierr)

  comm_time = MPI_Wtime()-t0

  ! Copy received y-halos to each GPU.
  ! Fill local periodic x and z ghosts.
  t0 = MPI_Wtime()

  do d = 1,ng
    call fill_ghosts( &
      slab(d)%p,slab(d)%recvlo,slab(d)%recvhi, &
      slab(d)%ny,slab(d)%dev)
  end do

  unpack_time = MPI_Wtime()-t0

  ! Second-order centered finite differences.
  t0 = MPI_Wtime()

  do d = 1,ng
    call compute_gradient( &
      slab(d)%p,slab(d)%gx,slab(d)%gy,slab(d)%gz, &
      slab(d)%ny,slab(d)%dev)
  end do

  fdm_time = MPI_Wtime()-t0
  numerical_time = MPI_Wtime()-numerical_start

  ! Per-GPU norm reductions first, then MPI reductions.
  t0 = MPI_Wtime()

  linf = 0.0_real64
  l1 = 0.0_real64
  l2 = 0.0_real64
  exact2 = 0.0_real64

  do d = 1,ng
    call calculate_norms( &
      slab(d)%gx,slab(d)%gy,slab(d)%gz, &
      slab(d)%ny,slab(d)%offset,slab(d)%dev, &
      di,d1,d2,de)

    linf = max(linf,di)
    l1 = l1+d1
    l2 = l2+d2
    exact2 = exact2+de
  end do

  norm_time = MPI_Wtime()-t0

  call MPI_Allreduce(MPI_IN_PLACE,linf,1, &
       MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)

  call MPI_Allreduce(MPI_IN_PLACE,l1,1, &
       MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

  call MPI_Allreduce(MPI_IN_PLACE,l2,1, &
       MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

  call MPI_Allreduce(MPI_IN_PLACE,exact2,1, &
       MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

  rel2 = sqrt(l2/exact2)
  expected = abs(1.0_real64-sin(twopi*h)/(twopi*h))

  l1 = l1*h**3
  l2 = sqrt(l2*h**3)

  application_time = MPI_Wtime()-start_time

  local_times = [init_time,pack_time,comm_time,unpack_time, &
                 fdm_time,norm_time,numerical_time,application_time]

  call MPI_Reduce(local_times,global_times,8, &
       MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)

  if (rank == 0) then
    print '(a)', '=== 3D gradient: MPI + OpenMP GPU ==='
    print '(a)', 'Decomposition = Y'
    print '(a,i0)', 'N = ',n
    print '(a,i0)', 'MPI_ranks = ',nranks
    print '(a,i0)', 'GPUs_per_rank = ',ng
    print '(a,i0)', 'GPUs = ',ng*nranks
    print '(a,es24.16)', 'dx = ',h
    print '(a,es24.16)', 'L_inf = ',linf
    print '(a,es24.16)', 'L1 = ',l1
    print '(a,es24.16)', 'L2 = ',l2
    print '(a,es24.16)', 'Relative_L2 = ',rel2
    print '(a,es24.16)', 'Expected_Relative_L2 = ',expected
    print '(a,es16.8)', 'Initialization_seconds = ',global_times(1)
    print '(a,es16.8)', 'Pack_seconds = ',global_times(2)
    print '(a,es16.8)', 'Communication_seconds = ',global_times(3)
    print '(a,es16.8)', 'Unpack_seconds = ',global_times(4)
    print '(a,es16.8)', 'FDM_seconds = ',global_times(5)
    print '(a,es16.8)', 'Norm_seconds = ',global_times(6)
    print '(a,es16.8)', 'Numerical_seconds = ',global_times(7)
    print '(a,es16.8)', 'Application_seconds = ',global_times(8)
  end if

  if (abs(rel2-expected) > 1.0e-10_real64) then
    if (rank == 0) print '(a)', 'Verification: FAIL'
    call MPI_Abort(MPI_COMM_WORLD,5,ierr)
  end if

  if (rank == 0) print '(a)', 'Verification: PASS'

  do d = 1,ng
    call exit_device( &
      slab(d)%p,slab(d)%gx,slab(d)%gy,slab(d)%gz, &
      slab(d)%sendlo,slab(d)%sendhi, &
      slab(d)%recvlo,slab(d)%recvhi,slab(d)%dev)
  end do

  call MPI_Finalize(ierr)

contains

  subroutine enter_device(p,gx,gy,gz,slo,shi,rlo,rhi,dev)
    real(real64), allocatable, intent(inout) :: p(:,:,:)
    real(real64), allocatable, intent(inout) :: gx(:,:,:),gy(:,:,:),gz(:,:,:)
    real(real64), allocatable, intent(inout) :: slo(:,:),shi(:,:)
    real(real64), allocatable, intent(inout) :: rlo(:,:),rhi(:,:)
    integer, intent(in) :: dev

    !$omp target enter data &
    !$omp& map(alloc:p,gx,gy,gz,slo,shi,rlo,rhi) device(dev)
  end subroutine

  subroutine exit_device(p,gx,gy,gz,slo,shi,rlo,rhi,dev)
    real(real64), allocatable, intent(inout) :: p(:,:,:)
    real(real64), allocatable, intent(inout) :: gx(:,:,:),gy(:,:,:),gz(:,:,:)
    real(real64), allocatable, intent(inout) :: slo(:,:),shi(:,:)
    real(real64), allocatable, intent(inout) :: rlo(:,:),rhi(:,:)
    integer, intent(in) :: dev

    !$omp target exit data &
    !$omp& map(delete:p,gx,gy,gz,slo,shi,rlo,rhi) device(dev)
  end subroutine

  subroutine initialize_pressure(p,ny,yo,dev)
    real(real64), allocatable, intent(inout) :: p(:,:,:)
    integer, intent(in) :: ny,yo,dev
    integer :: i,j,k
    real(real64) :: x,y,z

    !$omp target teams distribute parallel do collapse(3) &
    !$omp& private(x,y,z) device(dev)
    do k = 2,n+1
      do j = 2,ny+1
        do i = 2,n+1
          x = (real(i,real64)-1.5_real64)*h
          y = (real(yo+j,real64)-1.5_real64)*h
          z = (real(k,real64)-1.5_real64)*h
          p(i,j,k) = sin(twopi*x)*cos(twopi*y)*sin(twopi*z)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine pack_faces(p,slo,shi,ny,dev)
    real(real64), allocatable, intent(inout) :: p(:,:,:)
    real(real64), allocatable, intent(inout) :: slo(:,:),shi(:,:)
    integer, intent(in) :: ny,dev
    integer :: i,k

    !$omp target teams distribute parallel do collapse(2) device(dev)
    do k = 2,n+1
      do i = 2,n+1
        slo(i-1,k-1) = p(i,2,k)
        shi(i-1,k-1) = p(i,ny+1,k)
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target update from(slo,shi) device(dev)
  end subroutine

  subroutine fill_ghosts(p,rlo,rhi,ny,dev)
    real(real64), allocatable, intent(inout) :: p(:,:,:)
    real(real64), allocatable, intent(inout) :: rlo(:,:),rhi(:,:)
    integer, intent(in) :: ny,dev
    integer :: i,j,k

    !$omp target update to(rlo,rhi) device(dev)

    ! Y ghosts: communicated between GPUs and MPI ranks.
    !$omp target teams distribute parallel do collapse(2) device(dev)
    do k = 2,n+1
      do i = 2,n+1
        p(i,1,k) = rlo(i-1,k-1)
        p(i,ny+2,k) = rhi(i-1,k-1)
      end do
    end do
    !$omp end target teams distribute parallel do

    ! X ghosts: periodic, entirely local.
    !$omp target teams distribute parallel do collapse(2) device(dev)
    do k = 2,n+1
      do j = 2,ny+1
        p(1,j,k) = p(n+1,j,k)
        p(n+2,j,k) = p(2,j,k)
      end do
    end do
    !$omp end target teams distribute parallel do

    ! Z ghosts: periodic, entirely local.
    !$omp target teams distribute parallel do collapse(2) device(dev)
    do j = 2,ny+1
      do i = 2,n+1
        p(i,j,1) = p(i,j,n+1)
        p(i,j,n+2) = p(i,j,2)
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine compute_gradient(p,gx,gy,gz,ny,dev)
    real(real64), allocatable, intent(inout) :: p(:,:,:)
    real(real64), allocatable, intent(inout) :: gx(:,:,:),gy(:,:,:),gz(:,:,:)
    integer, intent(in) :: ny,dev
    integer :: i,j,k

    !$omp target teams distribute parallel do collapse(3) device(dev)
    do k = 2,n+1
      do j = 2,ny+1
        do i = 2,n+1
          gx(i-1,j-1,k-1) = &
             (p(i+1,j,k)-p(i-1,j,k))/(2.0_real64*h)

          gy(i-1,j-1,k-1) = &
             (p(i,j+1,k)-p(i,j-1,k))/(2.0_real64*h)

          gz(i-1,j-1,k-1) = &
             (p(i,j,k+1)-p(i,j,k-1))/(2.0_real64*h)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine calculate_norms(gx,gy,gz,ny,yo,dev, &
                             linf_out,l1_out,l2_out,exact_out)
    real(real64), allocatable, intent(inout) :: gx(:,:,:),gy(:,:,:),gz(:,:,:)
    integer, intent(in) :: ny,yo,dev
    real(real64), intent(out) :: linf_out,l1_out,l2_out,exact_out
    integer :: i,j,k
    real(real64) :: x,y,z,ex,ey,ez
    real(real64) :: linf_local,l1_local,l2_local,exact_local

    linf_local = 0.0_real64
    l1_local = 0.0_real64
    l2_local = 0.0_real64
    exact_local = 0.0_real64

    !$omp target teams distribute parallel do collapse(3) &
    !$omp& private(x,y,z,ex,ey,ez) &
    !$omp& reduction(max:linf_local) &
    !$omp& reduction(+:l1_local,l2_local,exact_local) device(dev)
    do k = 1,n
      do j = 1,ny
        do i = 1,n
          x = (real(i,real64)-0.5_real64)*h
          y = (real(yo+j,real64)-0.5_real64)*h
          z = (real(k,real64)-0.5_real64)*h

          ex = twopi*cos(twopi*x)*cos(twopi*y)*sin(twopi*z)
          ey = -twopi*sin(twopi*x)*sin(twopi*y)*sin(twopi*z)
          ez = twopi*sin(twopi*x)*cos(twopi*y)*cos(twopi*z)

          linf_local = max(linf_local, sqrt( &
             (gx(i,j,k)-ex)**2 + &
             (gy(i,j,k)-ey)**2 + &
             (gz(i,j,k)-ez)**2))

          l1_local = l1_local + sqrt( &
             (gx(i,j,k)-ex)**2 + &
             (gy(i,j,k)-ey)**2 + &
             (gz(i,j,k)-ez)**2)

          l2_local = l2_local + &
             (gx(i,j,k)-ex)**2 + &
             (gy(i,j,k)-ey)**2 + &
             (gz(i,j,k)-ez)**2

          exact_local = exact_local + ex**2+ey**2+ez**2
        end do
      end do
    end do
    !$omp end target teams distribute parallel do

    linf_out = linf_local
    l1_out = l1_local
    l2_out = l2_local
    exact_out = exact_local
  end subroutine

end program gradient_CPU_MPI_OMP_GPU
