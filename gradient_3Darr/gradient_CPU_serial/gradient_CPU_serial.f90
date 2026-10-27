program pressure_gradient_serial
  implicit none

  integer, parameter :: n = 64
  integer :: i, j, k, im, ip, jm, jp, km, kp
  real(8) :: pi, h, x, y, z
  real(8) :: exact_x, exact_y, exact_z
  real(8) :: error_sum, reference_sum, relative_error, expected_error
  real(8) :: p(n,n,n), gx(n,n,n), gy(n,n,n), gz(n,n,n)

  pi = acos(-1.0d0)
  h = 1.0d0/dble(n)

  ! STEP 1: Initialize pressure from the analytical expression

  do k = 1, n
    z = dble(k-1)*h

    do j = 1, n
      y = dble(j-1)*h

      do i = 1, n
        x = dble(i-1)*h

        p(i,j,k) = sin(2.0d0*pi*x)*cos(2.0d0*pi*y)*sin(2.0d0*pi*z)

      end do
    end do
  end do

  ! STEP 2: Compute the gradient numerically using central differences
  
  do k = 1, n
    km = k-1
    kp = k+1
    if (k == 1) km = n
    if (k == n) kp = 1

    do j = 1, n
      jm = j-1
      jp = j+1
      if (j == 1) jm = n
      if (j == n) jp = 1

      do i = 1, n
        im = i-1
        ip = i+1
        if (i == 1) im = n
        if (i == n) ip = 1

        gx(i,j,k) = (p(ip,j,k)-p(im,j,k))/(2.0d0*h)
        gy(i,j,k) = (p(i,jp,k)-p(i,jm,k))/(2.0d0*h)
        gz(i,j,k) = (p(i,j,kp)-p(i,j,km))/(2.0d0*h)

      end do
    end do
  end do

  ! STEP 3: Compare the numerical gradient to the analytical derivative

  error_sum = 0.0d0
  reference_sum = 0.0d0

  do k = 1, n
    z = dble(k-1)*h

    do j = 1, n
      y = dble(j-1)*h

      do i = 1, n
        x = dble(i-1)*h

        exact_x =  2.0d0*pi*cos(2.0d0*pi*x)*cos(2.0d0*pi*y)*sin(2.0d0*pi*z)
        exact_y = -2.0d0*pi*sin(2.0d0*pi*x)*sin(2.0d0*pi*y)*sin(2.0d0*pi*z)
        exact_z =  2.0d0*pi*sin(2.0d0*pi*x)*cos(2.0d0*pi*y)*cos(2.0d0*pi*z)

        error_sum = error_sum + (gx(i,j,k)-exact_x)**2 &
                              + (gy(i,j,k)-exact_y)**2 &
                              + (gz(i,j,k)-exact_z)**2

        reference_sum = reference_sum + exact_x**2 + exact_y**2 + exact_z**2

      end do
    end do
  end do

  relative_error = sqrt(error_sum/reference_sum)

  expected_error = abs(1.0d0 - sin(2.0d0*pi*h)/(2.0d0*pi*h))

  print '(a,i0)', 'Grid size N = ', n
  print '(a,es14.6)', 'Relative L2 error = ', relative_error
  print '(a,es14.6)', 'Expected L2 error = ', expected_error

  if (abs(relative_error-expected_error) < 1.0d-10) then
    print '(a)', 'Verification: PASS'
  else
    print '(a)', 'Verification: FAIL'
    error stop 1
  end if

end program pressure_gradient_serial
