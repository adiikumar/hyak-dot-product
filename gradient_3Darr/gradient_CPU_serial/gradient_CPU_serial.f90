program gradient_CPU_serial
  use iso_fortran_env, only: real64, int64
  implicit none

  integer :: n, ios, i, j, k, stat
  integer(int64) :: clock_start, clock_end, clock_rate
  character(len=64) :: arg

  real(real64), allocatable :: p(:,:,:)
  real(real64), allocatable :: gx(:,:,:), gy(:,:,:), gz(:,:,:)

  real(real64) :: h, twopi, x, y, z
  real(real64) :: ex, ey, ez
  real(real64) :: linf, l1, l2, exact2
  real(real64) :: relative_l2, expected
  real(real64) :: t0, t1, t2, t3
  real(real64) :: fdm_seconds, numerical_seconds, application_seconds

  n = 256

  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg)
    read(arg, *, iostat=ios) n
    if (ios /= 0) stop 1
  end if

  if (n < 4) stop 1

  h = 1.0_real64 / real(n, real64)
  twopi = 2.0_real64 * acos(-1.0_real64)

  call system_clock(count_rate=clock_rate)

  call system_clock(clock_start)
  t0 = real(clock_start, real64) / real(clock_rate, real64)

  allocate(p(n+2,n+2,n+2), &
           gx(n,n,n), gy(n,n,n), gz(n,n,n), stat=stat)

  if (stat /= 0) stop 'Allocation failed'

  ! Physical cells: 2:N+1
  ! Periodic ghosts: 1 and N+2

  do k = 2,n+1
    z = (real(k,real64)-1.5_real64)*h
    do j = 2,n+1
      y = (real(j,real64)-1.5_real64)*h
      do i = 2,n+1
        x = (real(i,real64)-1.5_real64)*h
        p(i,j,k) = sin(twopi*x)*cos(twopi*y)*sin(twopi*z)
      end do
    end do
  end do

  call system_clock(clock_end)
  t1 = real(clock_end,real64)/real(clock_rate,real64)

  ! Periodic x ghosts.
  do k = 2,n+1
    do j = 2,n+1
      p(1,j,k) = p(n+1,j,k)
      p(n+2,j,k) = p(2,j,k)
    end do
  end do

  ! Periodic y ghosts.
  do k = 2,n+1
    do i = 2,n+1
      p(i,1,k) = p(i,n+1,k)
      p(i,n+2,k) = p(i,2,k)
    end do
  end do

  ! Periodic z ghosts.
  do j = 2,n+1
    do i = 2,n+1
      p(i,j,1) = p(i,j,n+1)
      p(i,j,n+2) = p(i,j,2)
    end do
  end do

  call system_clock(clock_end)
  t2 = real(clock_end,real64)/real(clock_rate,real64)

  ! Second-order central difference.
  do k = 2,n+1
    do j = 2,n+1
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

  call system_clock(clock_end)
  t3 = real(clock_end,real64)/real(clock_rate,real64)

  fdm_seconds = t3-t2
  numerical_seconds = t3-t1

  linf = 0.0_real64
  l1 = 0.0_real64
  l2 = 0.0_real64
  exact2 = 0.0_real64

  do k = 1,n
    z = (real(k,real64)-0.5_real64)*h
    do j = 1,n
      y = (real(j,real64)-0.5_real64)*h
      do i = 1,n

        x = (real(i,real64)-0.5_real64)*h

        ex = twopi*cos(twopi*x)*cos(twopi*y)*sin(twopi*z)
        ey = -twopi*sin(twopi*x)*sin(twopi*y)*sin(twopi*z)
        ez = twopi*sin(twopi*x)*cos(twopi*y)*cos(twopi*z)

        linf = max(linf, sqrt( &
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

  relative_l2 = sqrt(l2/exact2)
  expected = abs(1.0_real64-sin(twopi*h)/(twopi*h))

  l1 = l1*h**3
  l2 = sqrt(l2*h**3)

  call system_clock(clock_end)
  application_seconds = &
    real(clock_end,real64)/real(clock_rate,real64)-t0

  print '(a)', '=== Cell-centered 3D gradient: SERIAL CPU ==='
  print '(a,i0)', 'N = ',n
  print '(a,es24.16)', 'dx = ',h
  print '(a,es24.16)', 'L_inf = ',linf
  print '(a,es24.16)', 'L1 = ',l1
  print '(a,es24.16)', 'L2 = ',l2
  print '(a,es24.16)', 'Relative_L2 = ',relative_l2
  print '(a,es24.16)', 'Expected_Relative_L2 = ',expected
  print '(a,es16.8)', 'FDM_seconds = ',fdm_seconds
  print '(a,es16.8)', 'Numerical_seconds = ',numerical_seconds
  print '(a,es16.8)', 'Application_seconds = ',application_seconds

  if (abs(relative_l2-expected) < 1.0e-10_real64) then
    print '(a)', 'Verification: PASS'
  else
    print '(a)', 'Verification: FAIL'
    stop 2
  end if

  deallocate(p,gx,gy,gz)

end program gradient_CPU_serial
