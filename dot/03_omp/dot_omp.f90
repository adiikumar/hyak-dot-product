! dot_omp.f90 : dot product with OpenMP directives on the do loop.
! Thread count is read from the command line (1, 2, ...).
program dot_omp
  use omp_lib
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: s, expected, t0, t1
  integer :: i, nthreads, used
  character(len=16) :: arg

  nthreads = 1
  if (command_argument_count() > 0) then
     call get_command_argument(1, arg)
     read(arg, *) nthreads
  end if
  call omp_set_num_threads(nthreads)

  allocate(a(n), b(n))

  !$omp parallel do
  do i = 1, n
     a(i) = real(i - 1, dp)
     b(i) = 1.0_dp
  end do
  !$omp end parallel do

  s = 0.0_dp
  used = 0
  t0 = omp_get_wtime()
  !$omp parallel
  !$omp single
  used = omp_get_num_threads()
  !$omp end single
  !$omp do reduction(+:s)
  do i = 1, n
     s = s + a(i) * b(i)
  end do
  !$omp end do
  !$omp end parallel
  t1 = omp_get_wtime()

  expected = real(n, dp) * real(n - 1, dp) / 2.0_dp
  print '(A,I0,A,I0)', 'Threads requested = ', nthreads, ', used = ', used
  print '(A,F18.1)', 'OpenMP dot product = ', s
  print '(A,F18.1)', 'Expected           = ', expected
  if (s == expected) then
     print '(A,F10.6,A)', 'Verification: PASS   (time ', t1 - t0, ' s)'
  else
     print '(A,F10.6,A)', 'Verification: FAIL   (time ', t1 - t0, ' s)'
  end if

  deallocate(a, b)
end program dot_omp
