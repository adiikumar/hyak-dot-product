
! gemm_serial_blas.f90 : serial matrix multiplication using BLAS DGEMM
program gemm_serial_blas
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: n, i, j, k
  integer(kind=8) :: tick0, tick1, rate

  real(dp), allocatable :: a(:,:), b(:,:), c(:,:)
  real(dp) :: s1, s2, expected, max_err, err
  real(dp) :: time_s, gflops

  character(len=32) :: arg
  external :: dgemm

  ! Matrix size (default N = 1024)
  n = 1024
  if (command_argument_count() > 0) then
     call get_command_argument(1, arg)
     read(arg, *) n
  end if

  if (n < 1) error stop 'Matrix dimension must be positive'

  allocate(a(n,n), b(n,n), c(n,n))

  ! Initialize matrices using 0-based index values
  ! A(i,k) = (i-1) + (k-1)
  ! B(k,j) = (k-1) + 2*(j-1)

  do k = 1, n
     do i = 1, n
        a(i,k) = real(i-1,dp) + real(k-1,dp)
     end do
  end do

  do j = 1, n
     do k = 1, n
        b(k,j) = real(k-1,dp) + 2.0_dp*real(j-1,dp)
     end do
  end do

  ! Matrix multiplication C = A * B using BLAS
  c = 0.0_dp

  call system_clock(tick0, rate)

  call dgemm('N', 'N', n, n, n, &
             1.0_dp, a, n, b, n, 0.0_dp, c, n)

  call system_clock(tick1)

  time_s = real(tick1-tick0,dp) / real(rate,dp)

  ! Analytical verification
  ! sum(k) and sum(k^2), for k = 0 to N-1

  s1 = real(n,dp)*real(n-1,dp)/2.0_dp

  s2 = real(n,dp)*real(n-1,dp) * &
       (2.0_dp*real(n,dp)-1.0_dp)/6.0_dp

  max_err = 0.0_dp

  do j = 1, n
     do i = 1, n

        expected = s2 + &
          (real(i-1,dp)+2.0_dp*real(j-1,dp))*s1 + &
          2.0_dp*real(n,dp)*real(i-1,dp)*real(j-1,dp)

        err = abs(c(i,j)-expected) / &
              max(1.0_dp,abs(expected))

        max_err = max(max_err,err)

     end do
  end do

  ! Computational performance
  if (time_s > 0.0_dp) then
     gflops = 2.0_dp*real(n,dp)**3 / &
              (time_s*1.0e9_dp)
  else
     gflops = 0.0_dp
  end if

  print '(A,I0)', 'Matrix dimension N = ', n
  print '(A,ES18.8)', 'C(1,1) = ', c(1,1)
  print '(A,ES12.4)', 'Maximum relative error = ', max_err

  if (max_err < 1.0e-10_dp) then
     print *, 'Verification: PASS'
  else
     print *, 'Verification: FAIL'
  end if

  print '(A,F12.6,A)', 'DGEMM time = ', time_s, ' s'
  print '(A,F12.3)', 'DGEMM GFLOP/s = ', gflops

  deallocate(a,b,c)

end program gemm_serial_blas
