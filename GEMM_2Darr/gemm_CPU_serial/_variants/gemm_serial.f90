program gemm_serial_loop
  use iso_fortran_env, only: int64
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: n, i, j, k, ios
  integer(int64) :: tick0, tick1, rate
  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)
  real(dp) :: s, s1, s2, expected, error, max_error, seconds
  character(len=32) :: arg

  n = 2048
  if (command_argument_count() >= 1) then
     call get_command_argument(1, arg)
     read(arg, *, iostat=ios) n
     if (ios /= 0) error stop 'Invalid N'
  end if
  if (n < 1) error stop 'N must be positive'

  allocate(a(n,n), b(n,n), c(n,n))

  ! Initialize A.
  do k = 1, n
     do i = 1, n
        a(i,k) = real(i-1,dp) + real(k-1,dp)
     end do
  end do

  ! Initialize B.
  do j = 1, n
     do k = 1, n
        b(k,j) = real(k-1,dp) + 2.0_dp*real(j-1,dp)
     end do
  end do

  call system_clock(tick0, rate)

  ! Explicit matrix multiplication.
  do j = 1, n
     do i = 1, n
        s = 0.0_dp
        do k = 1, n
           s = s + a(i,k)*b(k,j)
        end do
        c(i,j) = s
     end do
  end do

  call system_clock(tick1)
  seconds = real(tick1-tick0,dp) / real(rate,dp)

  ! Analytical verification.
  s1 = real(n,dp)*real(n-1,dp)/2.0_dp
  s2 = real(n,dp)*real(n-1,dp)*(2.0_dp*real(n,dp)-1.0_dp)/6.0_dp

  max_error = 0.0_dp
  do j = 1, n
     do i = 1, n
        expected = s2 + (real(i-1,dp)+2.0_dp*real(j-1,dp))*s1 &
                   + 2.0_dp*real(n,dp)*real(i-1,dp)*real(j-1,dp)
        error = abs(c(i,j)-expected)/max(1.0_dp,abs(expected))
        max_error = max(max_error,error)
     end do
  end do

  print '(A,I0)', 'Matrix dimension N = ', n
  print '(A,ES18.8)', 'C(1,1) = ', c(1,1)
  print '(A,ES12.4)', 'Maximum relative error = ', max_error

  if (max_error < 1.0e-10_dp) then
     print *, 'Verification: PASS'
  else
     print *, 'Verification: FAIL'
     error stop 1
  end if

  print '(A,F12.6,A)', 'Loop GEMM time = ', seconds, ' s'
  print '(A,F12.3)', 'Loop GEMM GFLOP/s = ', &
       2.0_dp*real(n,dp)**3/(seconds*1.0e9_dp)

end program gemm_serial_loop
