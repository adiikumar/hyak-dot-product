! dot_serial.f90 : serial dot product of two vectors
program dot_serial
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: s, expected
  integer :: i

  allocate(a(n), b(n))

  ! a(i) = i-1 (0-based values), b(i) = 1  ->  exact answer N(N-1)/2
  do i = 1, n
     a(i) = real(i - 1, dp)
     b(i) = 1.0_dp
  end do

  s = 0.0_dp
  do i = 1, n
     s = s + a(i) * b(i)
  end do

  expected = real(n, dp) * real(n - 1, dp) / 2.0_dp
  print '(A,F18.1)', 'Serial dot product = ', s
  print '(A,F18.1)', 'Expected           = ', expected
  if (s == expected) then
     print *, 'Verification: PASS'
  else
     print *, 'Verification: FAIL'
  end if

  deallocate(a, b)
end program dot_serial
