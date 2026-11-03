program gradient_CPU_MPI
  use iso_fortran_env, only: real64
  use mpi
  implicit none

  integer :: ierr, rank, nranks, n, ny, offset
  integer :: prev, next, q, rem, stat, ios
  integer :: i, j, k, status(MPI_STATUS_SIZE)
  character(len=64) :: arg

  real(real64), allocatable :: p(:,:,:)
  real(real64), allocatable :: gx(:,:,:), gy(:,:,:), gz(:,:,:)
  real(real64), allocatable :: sendlo(:,:), sendhi(:,:)
  real(real64), allocatable :: recvlo(:,:), recvhi(:,:)

  real(real64) :: h, twopi, x, y, z
  real(real64) :: ex, ey, ez
  real(real64) :: linf, l1, l2, exact2
  real(real64) :: global_linf, global_l1, global_l2, global_exact2
  real(real64) :: relative_l2, expected
  real(real64) :: t0, t1, t2, t3, t4
  real(real64) :: times(5), max_times(5)

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nranks,ierr)

  n = 2048

  if (command_argument_count() >= 1) then
    call get_command_argument(1,arg)
    read(arg,*,iostat=ios) n
    if (ios /= 0) n = -1
  end if

  if (n < 4 .or. nranks > n) then
    if (rank == 0) print *, 'Invalid N or MPI rank count'
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end if

  h = 1.0_real64/real(n,real64)
  twopi = 2.0_real64*acos(-1.0_real64)

  ! Y-slab decomposition.
  q = n/nranks
  rem = mod(n,nranks)

  ny = q
  if (rank < rem) ny = ny+1

  offset = rank*q + min(rank,rem)

  prev = modulo(rank-1,nranks)
  next = modulo(rank+1,nranks)

  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  t0 = MPI_Wtime()

  allocate(p(n+2,ny+2,n+2), &
           gx(n,ny,n), gy(n,ny,n), gz(n,ny,n), &
           sendlo(n,n), sendhi(n,n), &
           recvlo(n,n), recvhi(n,n), stat=stat)

  if (stat /= 0) then
    print *, 'Allocation failed on rank ',rank
    call MPI_Abort(MPI_COMM_WORLD,2,ierr)
  end if

  ! Physical: i=2:N+1, j=2:ny+1, k=2:N+1.
  ! Global y cell is offset+j-1.

  do k = 2,n+1
    z = (real(k,real64)-1.5_real64)*h
    do j = 2,ny+1
      y = (real(offset+j,real64)-1.5_real64)*h
      do i = 2,n+1
        x = (real(i,real64)-1.5_real64)*h
        p(i,j,k) = sin(twopi*x)*cos(twopi*y)*sin(twopi*z)
      end do
    end do
  end do

  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  t1 = MPI_Wtime()

  ! Pack physical y-boundary planes.
  do k = 2,n+1
    do i = 2,n+1
      sendlo(i-1,k-1) = p(i,2,k)
      sendhi(i-1,k-1) = p(i,ny+1,k)
    end do
  end do

  ! Periodic y-halo exchange.
  call MPI_Sendrecv( &
    sendhi,n*n,MPI_DOUBLE_PRECISION,next,10, &
    recvlo,n*n,MPI_DOUBLE_PRECISION,prev,10, &
    MPI_COMM_WORLD,status,ierr)

  call MPI_Sendrecv( &
    sendlo,n*n,MPI_DOUBLE_PRECISION,prev,11, &
    recvhi,n*n,MPI_DOUBLE_PRECISION,next,11, &
    MPI_COMM_WORLD,status,ierr)

  do k = 2,n+1
    do i = 2,n+1
      p(i,1,k) = recvlo(i-1,k-1)
      p(i,ny+2,k) = recvhi(i-1,k-1)
    end do
  end do

  ! Local periodic x ghosts.
  do k = 2,n+1
    do j = 2,ny+1
      p(1,j,k) = p(n+1,j,k)
      p(n+2,j,k) = p(2,j,k)
    end do
  end do

  ! Local periodic z ghosts.
  do j = 2,ny+1
    do i = 2,n+1
      p(i,j,1) = p(i,j,n+1)
      p(i,j,n+2) = p(i,j,2)
    end do
  end do

  t2 = MPI_Wtime()

  ! Central differences.
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

  t3 = MPI_Wtime()

  linf = 0.0_real64
  l1 = 0.0_real64
  l2 = 0.0_real64
  exact2 = 0.0_real64

  do k = 1,n
    z = (real(k,real64)-0.5_real64)*h
    do j = 1,ny
      y = (real(offset+j,real64)-0.5_real64)*h
      do i = 1,n

        x = (real(i,real64)-0.5_real64)*h

        ex = twopi*cos(twopi*x)*cos(twopi*y)*sin(twopi*z)
        ey = -twopi*sin(twopi*x)*sin(twopi*y)*sin(twopi*z)
        ez = twopi*sin(twopi*x)*cos(twopi*y)*cos(twopi*z)

        linf = max(linf,sqrt( &
          (gx(i,j,k)-ex)**2 + &
          (gy(i,j,k)-ey)**2 + &
          (gz(i,j,k)-ez)**2))

        l1 = l1 + sqrt( &
          (gx(i,j,k)-ex)**2 + &
          (gy(i,j,k)-ey)**2 + &
          (gz(i,j,k)-ez)**2)

        l2 = l2 + &
          (gx(i,j,k)-ex)**2 + &
          (gy(i,j,k)-ey)**2 + &
          (gz(i,j,k)-ez)**2

        exact2 = exact2 + ex**2 + ey**2 + ez**2

      end do
    end do
  end do

  ! MPI global error norms.
  call MPI_Allreduce(linf,global_linf,1, &
    MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)

  call MPI_Allreduce(l1,global_l1,1, &
    MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

  call MPI_Allreduce(l2,global_l2,1, &
    MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

  call MPI_Allreduce(exact2,global_exact2,1, &
    MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)

  relative_l2 = sqrt(global_l2/global_exact2)
  expected = abs(1.0_real64-sin(twopi*h)/(twopi*h))

  global_l1 = global_l1*h**3
  global_l2 = sqrt(global_l2*h**3)

  times(1) = t3-t2
  times(2) = t3-t1
  t4 = MPI_Wtime()
  times(3) = t4-t0
  times(4) = t1-t0
  times(5) = t4-t3

  call MPI_Reduce(times,max_times,5, &
    MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)

  if (rank == 0) then
    print '(a)', '=== Cell-centered 3D gradient: MPI CPU ==='
    print '(a)', 'Decomposition = Y'
    print '(a,i0)', 'N = ',n
    print '(a,i0)', 'MPI_ranks = ',nranks
    print '(a,es24.16)', 'dx = ',h
    print '(a,es24.16)', 'L_inf = ',global_linf
    print '(a,es24.16)', 'L1 = ',global_l1
    print '(a,es24.16)', 'L2 = ',global_l2
    print '(a,es24.16)', 'Relative_L2 = ',relative_l2
    print '(a,es24.16)', 'Expected_Relative_L2 = ',expected
    print '(a,es16.8)', 'FDM_seconds = ',max_times(1)
    print '(a,es16.8)', 'Numerical_seconds = ',max_times(2)
    print '(a,es16.8)', 'Application_seconds = ',max_times(3)
    print '(a,es16.8)', 'Initialization_seconds = ',max_times(4)
    print '(a,es16.8)', 'Norm_seconds = ',max_times(5)
  end if

  if (abs(relative_l2-expected) < 1.0e-10_real64) then
    if (rank == 0) print '(a)', 'Verification: PASS'
  else
    if (rank == 0) print '(a)', 'Verification: FAIL'
    call MPI_Abort(MPI_COMM_WORLD,3,ierr)
  end if

  deallocate(p,gx,gy,gz,sendlo,sendhi,recvlo,recvhi)

  call MPI_Finalize(ierr)

end program gradient_CPU_MPI
