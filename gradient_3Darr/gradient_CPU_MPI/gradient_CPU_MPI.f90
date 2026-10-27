program pressure_gradient_mpi
  use mpi
  implicit none

  integer, parameter :: n = 64

  integer :: ierr, rank, nprocs, status(MPI_STATUS_SIZE)
  integer :: nloc, base, rem, istart
  integer :: left, right
  integer :: i, j, k
  real(8) :: pi, h, x, y, z, wave
  real(8) :: exact_x, exact_y, exact_z
  real(8) :: error_local, error_global
  real(8) :: reference_local, reference_global
  real(8) :: relative_error, expected_error

  real(8), allocatable :: p(:,:,:)
  real(8), allocatable :: gx(:,:,:), gy(:,:,:), gz(:,:,:)
  real(8), allocatable :: send_left(:,:), send_right(:,:)
  real(8), allocatable :: recv_left(:,:), recv_right(:,:)

  ! STEP 1: Initialize MPI

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nprocs,ierr)

  if (nprocs > n) then
    if (rank == 0) print *, 'Too many MPI ranks'
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end if

  pi = acos(-1.0d0)
  wave = 2.0d0*pi
  h = 1.0d0/dble(n)

  ! STEP 2: Divide the domain along x

  base = n/nprocs
  rem = mod(n,nprocs)

  nloc = base
  if (rank < rem) nloc = nloc + 1

  istart = rank*base + min(rank,rem)

  ! Identify periodic neighboring MPI ranks

  left  = modulo(rank-1,nprocs)
  right = modulo(rank+1,nprocs)

  ! STEP 3: Allocate local arrays with ghost cells

  allocate(p(0:nloc+1,0:n+1,0:n+1))
  allocate(gx(nloc,n,n),gy(nloc,n,n),gz(nloc,n,n))

  allocate(send_left(n,n),send_right(n,n))
  allocate(recv_left(n,n),recv_right(n,n))

  p = 0.0d0

  ! STEP 4: Initialize the local pressure field

  do k = 1, n
    z = dble(k-1)*h

    do j = 1, n
      y = dble(j-1)*h

      do i = 1, nloc
        x = dble(istart+i-1)*h

        p(i,j,k) = sin(wave*x)*cos(wave*y)*sin(wave*z)

      end do
    end do
  end do

  ! STEP 5: Pack x-boundary planes into contiguous buffers

  do k = 1, n
    do j = 1, n

      send_left(j,k)  = p(1,j,k)
      send_right(j,k) = p(nloc,j,k)

    end do
  end do

  ! STEP 6: Exchange boundary planes with MPI

  ! Send rightmost interior plane to the right neighbor.
  ! Receive the left neighbor's rightmost plane.

  call MPI_Sendrecv(send_right,n*n,MPI_DOUBLE_PRECISION, &
                    right,10, &
                    recv_left,n*n,MPI_DOUBLE_PRECISION, &
                    left,10,MPI_COMM_WORLD,status,ierr)

  ! Send leftmost interior plane to the left neighbor.
  ! Receive the right neighbor's leftmost plane.

  call MPI_Sendrecv(send_left,n*n,MPI_DOUBLE_PRECISION, &
                    left,11, &
                    recv_right,n*n,MPI_DOUBLE_PRECISION, &
                    right,11,MPI_COMM_WORLD,status,ierr)

  ! STEP 7: Copy received values into x-ghost cells

  do k = 1, n
    do j = 1, n

      p(0,j,k)      = recv_left(j,k)
      p(nloc+1,j,k) = recv_right(j,k)

    end do
  end do

  ! STEP 8: Apply local periodic boundaries in y and z

  ! Y-direction

  do k = 1, n
    do i = 1, nloc

      p(i,0,k)   = p(i,n,k)
      p(i,n+1,k) = p(i,1,k)

    end do
  end do

  ! Z-direction

  do j = 1, n
    do i = 1, nloc

      p(i,j,0)   = p(i,j,n)
      p(i,j,n+1) = p(i,j,1)

    end do
  end do

  ! STEP 9: Compute numerical gradient using FDM

  do k = 1, n
    do j = 1, n
      do i = 1, nloc

        gx(i,j,k) = (p(i+1,j,k)-p(i-1,j,k))/(2.0d0*h)

        gy(i,j,k) = (p(i,j+1,k)-p(i,j-1,k))/(2.0d0*h)

        gz(i,j,k) = (p(i,j,k+1)-p(i,j,k-1))/(2.0d0*h)

      end do
    end do
  end do

  ! STEP 10: Compare against the analytical gradient

  error_local = 0.0d0
  reference_local = 0.0d0

  do k = 1, n
    z = dble(k-1)*h

    do j = 1, n
      y = dble(j-1)*h

      do i = 1, nloc
        x = dble(istart+i-1)*h

        exact_x = wave*cos(wave*x)*cos(wave*y)*sin(wave*z)
        exact_y = -wave*sin(wave*x)*sin(wave*y)*sin(wave*z)
        exact_z = wave*sin(wave*x)*cos(wave*y)*cos(wave*z)

        error_local = error_local + &
          (gx(i,j,k)-exact_x)**2 + &
          (gy(i,j,k)-exact_y)**2 + &
          (gz(i,j,k)-exact_z)**2

        reference_local = reference_local + &
          exact_x**2 + exact_y**2 + exact_z**2

      end do
    end do
  end do

  ! STEP 11: Combine error sums across MPI ranks

  call MPI_Reduce(error_local,error_global,1, &
                  MPI_DOUBLE_PRECISION,MPI_SUM,0, &
                  MPI_COMM_WORLD,ierr)

  call MPI_Reduce(reference_local,reference_global,1, &
                  MPI_DOUBLE_PRECISION,MPI_SUM,0, &
                  MPI_COMM_WORLD,ierr)

  ! STEP 12: Global verification on rank 0

  if (rank == 0) then

    relative_error = sqrt(error_global/reference_global)

    expected_error = abs(1.0d0-sin(wave*h)/(wave*h))

    print *, 'Grid N = ', n
    print *, 'MPI ranks = ', nprocs
    print *, 'Relative L2 error = ', relative_error
    print *, 'Expected L2 error = ', expected_error

    if (abs(relative_error-expected_error) < 1.0d-10) then
      print *, 'Verification: PASS'
    else
      print *, 'Verification: FAIL'
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    end if

  end if

  deallocate(p,gx,gy,gz)
  deallocate(send_left,send_right,recv_left,recv_right)

  call MPI_Finalize(ierr)

end program pressure_gradient_mpi
