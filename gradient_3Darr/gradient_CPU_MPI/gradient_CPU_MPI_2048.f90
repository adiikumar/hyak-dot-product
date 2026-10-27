program pressure_gradient_mpi_2048
  use mpi
  implicit none

  integer :: n, ierr, rank, nprocs, status(MPI_STATUS_SIZE)
  integer :: base, rem, nloc, istart, left, right
  integer :: i, j, k, q, ios, alloc_stat
  character(len=64) :: arg

  integer, parameter :: nphase = 12
  character(len=38), parameter :: phase_name(nphase) = [character(len=38) :: &
       'Allocate arrays', &
       'Initialize pressure', &
       'Pack x boundary planes', &
       'MPI_Sendrecv: left ghost', &
       'MPI_Sendrecv: right ghost', &
       'Unpack x ghost planes', &
       'Local periodic y/z faces', &
       'FDM gradient', &
       'Analytical error calculation', &
       'MPI_Reduce: global errors', &
       'Halo pipeline (pack+exchange+unpack)', &
       'Total time (excludes reporting)' ]

  real(8) :: pi, wave, h, x, y, z
  real(8) :: exact_x, exact_y, exact_z
  real(8) :: error_local, error_global
  real(8) :: reference_local, reference_global
  real(8) :: relative_error, expected_error

  real(8) :: t0, tstart
  real(8) :: dt(nphase), tmin(nphase), tmax(nphase), tsum(nphase)
  real(8) :: halo_fraction, fdm_fraction
  real(8) :: halo_fracs(3), fdm_fracs(3)
  real(8) :: send_gib, array_gib, bytes_per_face

  real(8), allocatable :: p(:,:,:)
  real(8), allocatable :: gx(:,:,:), gy(:,:,:), gz(:,:,:)
  real(8), allocatable :: send_left(:,:), send_right(:,:)
  real(8), allocatable :: recv_left(:,:), recv_right(:,:)

  ! STEP 1: Initialize MPI and read grid size

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nprocs,ierr)

  n = 2048

  if (command_argument_count() >= 1) then
    call get_command_argument(1,arg)
    read(arg,*,iostat=ios) n
    if (ios /= 0) n = -1
  end if

  if (n < 4 .or. nprocs > n .or. n > 46340) then
    if (rank == 0) then
      print *, 'ERROR: Require 4 <= N <= 46340 and ranks <= N'
    end if
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end if

  pi = acos(-1.0d0)
  wave = 2.0d0*pi
  h = 1.0d0/dble(n)

  ! STEP 2: Divide the domain into x-slabs

  base = n/nprocs
  rem = mod(n,nprocs)

  nloc = base
  if (rank < rem) nloc = nloc + 1

  istart = rank*base + min(rank,rem)

  left = modulo(rank-1,nprocs)
  right = modulo(rank+1,nprocs)

  dt = 0.0d0

  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  tstart = MPI_Wtime()

  ! STEP 3: Allocate arrays including ghost cells

  t0 = MPI_Wtime()

  allocate(p(0:nloc+1,0:n+1,0:n+1), &
           gx(nloc,n,n), gy(nloc,n,n), gz(nloc,n,n), &
           send_left(n,n), send_right(n,n), &
           recv_left(n,n), recv_right(n,n), stat=alloc_stat)

  if (alloc_stat /= 0) then
    print *, 'Allocation failed on rank ',rank
    call MPI_Abort(MPI_COMM_WORLD,2,ierr)
  end if

  dt(1) = MPI_Wtime()-t0

  ! STEP 4: Initialize pressure

  t0 = MPI_Wtime()

  p = 0.0d0

  do k = 1,n
    z = dble(k-1)*h
    do j = 1,n
      y = dble(j-1)*h
      do i = 1,nloc
        x = dble(istart+i-1)*h

        p(i,j,k) = sin(wave*x)*cos(wave*y)*sin(wave*z)
      end do
    end do
  end do

  dt(2) = MPI_Wtime()-t0

  ! STEP 5: Pack boundary planes

  t0 = MPI_Wtime()

  do k = 1,n
    do j = 1,n
      send_left(j,k) = p(1,j,k)
      send_right(j,k) = p(nloc,j,k)
    end do
  end do

  dt(3) = MPI_Wtime()-t0

  ! STEP 6: Exchange right face to fill left ghost

  t0 = MPI_Wtime()

  call MPI_Sendrecv(send_right,n*n,MPI_DOUBLE_PRECISION,right,10, &
                    recv_left,n*n,MPI_DOUBLE_PRECISION,left,10, &
                    MPI_COMM_WORLD,status,ierr)

  dt(4) = MPI_Wtime()-t0

  ! STEP 7: Exchange left face to fill right ghost

  t0 = MPI_Wtime()

  call MPI_Sendrecv(send_left,n*n,MPI_DOUBLE_PRECISION,left,11, &
                    recv_right,n*n,MPI_DOUBLE_PRECISION,right,11, &
                    MPI_COMM_WORLD,status,ierr)

  dt(5) = MPI_Wtime()-t0

  ! STEP 8: Unpack received ghost planes

  t0 = MPI_Wtime()

  do k = 1,n
    do j = 1,n
      p(0,j,k) = recv_left(j,k)
      p(nloc+1,j,k) = recv_right(j,k)
    end do
  end do

  dt(6) = MPI_Wtime()-t0

  ! STEP 9: Apply local periodic y and z boundaries

  t0 = MPI_Wtime()

  do k = 1,n
    do i = 1,nloc
      p(i,0,k) = p(i,n,k)
      p(i,n+1,k) = p(i,1,k)
    end do
  end do

  do j = 1,n
    do i = 1,nloc
      p(i,j,0) = p(i,j,n)
      p(i,j,n+1) = p(i,j,1)
    end do
  end do

  dt(7) = MPI_Wtime()-t0

  ! STEP 10: Calculate the finite-difference gradient

  t0 = MPI_Wtime()

  do k = 1,n
    do j = 1,n
      do i = 1,nloc

        gx(i,j,k) = (p(i+1,j,k)-p(i-1,j,k))/(2.0d0*h)

        gy(i,j,k) = (p(i,j+1,k)-p(i,j-1,k))/(2.0d0*h)

        gz(i,j,k) = (p(i,j,k+1)-p(i,j,k-1))/(2.0d0*h)

      end do
    end do
  end do

  dt(8) = MPI_Wtime()-t0

  ! STEP 11: Compare against the analytical gradient

  t0 = MPI_Wtime()

  error_local = 0.0d0
  reference_local = 0.0d0

  do k = 1,n
    z = dble(k-1)*h

    do j = 1,n
      y = dble(j-1)*h

      do i = 1,nloc
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

  dt(9) = MPI_Wtime()-t0

  ! STEP 12: Reduce global errors

  t0 = MPI_Wtime()

  call MPI_Reduce(error_local,error_global,1, &
                  MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(reference_local,reference_global,1, &
                  MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)

  dt(10) = MPI_Wtime()-t0

  ! STEP 13: Combined measurements

  dt(11) = sum(dt(3:6))
  dt(12) = MPI_Wtime()-tstart

  halo_fraction = 100.0d0*dt(11)/max(dt(12),tiny(1.0d0))
  fdm_fraction  = 100.0d0*dt(8)/max(dt(12),tiny(1.0d0))

  ! STEP 14: Collect timing statistics across ranks
  ! These reductions are excluded from the total time.

  call MPI_Reduce(dt,tmin,nphase,MPI_DOUBLE_PRECISION, &
                  MPI_MIN,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(dt,tsum,nphase,MPI_DOUBLE_PRECISION, &
                  MPI_SUM,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(dt,tmax,nphase,MPI_DOUBLE_PRECISION, &
                  MPI_MAX,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(halo_fraction,halo_fracs(1),1, &
                  MPI_DOUBLE_PRECISION,MPI_MIN,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(halo_fraction,halo_fracs(2),1, &
                  MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(halo_fraction,halo_fracs(3),1, &
                  MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(fdm_fraction,fdm_fracs(1),1, &
                  MPI_DOUBLE_PRECISION,MPI_MIN,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(fdm_fraction,fdm_fracs(2),1, &
                  MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)

  call MPI_Reduce(fdm_fraction,fdm_fracs(3),1, &
                  MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)

  ! STEP 15: Print results from rank 0

  if (rank == 0) then

    relative_error = sqrt(error_global/reference_global)
    expected_error = abs(1.0d0-sin(wave*h)/(wave*h))

    bytes_per_face = 8.0d0*dble(n)*dble(n)

    send_gib = 2.0d0*bytes_per_face*dble(nprocs) &
               /1073741824.0d0

    array_gib = 8.0d0*( &
           dble(nloc+2)*dble(n+2)*dble(n+2) + &
           3.0d0*dble(nloc)*dble(n)*dble(n) + &
           4.0d0*dble(n)*dble(n)) /1073741824.0d0

    print '(a)', '=== MPI 3D Periodic Pressure Gradient ==='

    print '(a,i0,a,i0,a,i0)', &
          'Global grid: ',n,' x ',n,' x ',n

    print '(a,i0)', 'MPI ranks: ',nprocs

    print '(a,i0,a,i0)', &
          'x-planes per rank: min=',base, &
          ', max=',base+min(1,rem)

    print '(a,f12.3,a)', &
          'One halo face: ',bytes_per_face/1048576.0d0,' MiB'

    print '(a,f12.3,a)', &
          'All-rank MPI sends: ',send_gib,' GiB (logical payload)'

    print '(a,f12.3,a)', &
          'Rank-0 array allocation: ',array_gib,' GiB (estimated)'

    print '(a)', &
          'MPI_Bcast: not used; pressure is initialized locally.'

    print '(a)', ''
    print '(a)', '=== Phase times across ranks (seconds) ==='

    print '(a38,3(1x,a12))', 'Phase','MIN','MEAN','MAX'

    do q = 1,nphase
      print '(a38,3(1x,f12.6))', &
            phase_name(q),tmin(q),tsum(q)/dble(nprocs),tmax(q)
    end do

    print '(a)', ''

    print '(a,3(1x,f9.3))', &
          'Halo fraction % (min/mean/max):', &
          halo_fracs(1),halo_fracs(2)/dble(nprocs),halo_fracs(3)

    print '(a,3(1x,f9.3))', &
          'FDM fraction % (min/mean/max):', &
          fdm_fracs(1),fdm_fracs(2)/dble(nprocs),fdm_fracs(3)

    print '(a,es20.12)', 'Relative L2 error = ',relative_error
    print '(a,es20.12)', 'Expected L2 error = ',expected_error

    if (abs(relative_error-expected_error) < 1.0d-10) then
      print *, 'Verification: PASS'
    else
      print *, 'Verification: FAIL'
      call MPI_Abort(MPI_COMM_WORLD,3,ierr)
    end if

    print '(a)', &
      'MPI_Sendrecv timings include time spent waiting for neighbors.'

    print '(a)', &
      'Phase MIN/MEAN/MAX values cannot simply be added to get total.'

  end if

  deallocate(p,gx,gy,gz)
  deallocate(send_left,send_right,recv_left,recv_right)

  call MPI_Finalize(ierr)

end program pressure_gradient_mpi_2048
