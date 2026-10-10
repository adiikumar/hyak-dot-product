! dot_gpu_only.f90 : GPU-only dot product, written from dot_hybrid.f90.
! One GPU does all the work, using OpenMP target offload (no CUDA).
! There is no MPI and there are no CPU OpenMP threads, so no CPU does any
! arithmetic. The CPU only starts the program, launches the GPU kernels,
! waits for them and prints, because a GPU cannot start a program by itself.
!
! Build:  module load nvhpc/24.5 gcc/11.2.0
!         nvfortran -O2 -mp=gpu -gpu=cc75,cc80,cc86,cc89,cc90 dot_gpu_only.f90 -o dot_gpu_only
! Run:    OMP_TARGET_OFFLOAD=MANDATORY ./dot_gpu_only
program dot_gpu_only
  use omp_lib
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 1000000
  integer :: ndev, i
  real(dp), allocatable :: a(:), b(:)
  real(dp) :: dot, expected, t0, t1
  logical :: onhost

  ! Prove that a target region really runs on the GPU, and stop if it does not
  ndev = omp_get_num_devices()
  onhost = .true.
  !$omp target map(from: onhost)
  onhost = omp_is_initial_device()
  !$omp end target
  if (onhost) then
     print '(A,I0,A)', 'Offload devices visible = ', ndev, ', target region ran on the HOST. Stopping, no GPU available.'
     stop 1
  end if
  print '(A,I0,A)', 'Offload devices visible = ', ndev, ', target region ran on the GPU'

  ! The arrays live on the GPU only. Host memory is reserved but never read or written.
  allocate(a(n), b(n))
  !$omp target enter data map(alloc: a, b)

  ! Same data as serial: a = global index (0-based), b = 1, filled by GPU threads
  !$omp target teams distribute parallel do
  do i = 1, n
     a(i) = real(i - 1, dp)
     b(i) = 1.0_dp
  end do
  !$omp end target teams distribute parallel do

  ! Dot product on the GPU: each GPU thread keeps a partial sum, the OpenMP
  ! reduction adds them, and the final number is copied back with the region
  dot = 0.0_dp
  t0 = omp_get_wtime()
  !$omp target teams distribute parallel do reduction(+:dot)
  do i = 1, n
     dot = dot + a(i) * b(i)
  end do
  !$omp end target teams distribute parallel do
  t1 = omp_get_wtime()

  expected = real(n, dp) * real(n - 1, dp) / 2.0_dp
  print '(A,F18.1)', 'GPU dot product = ', dot
  print '(A,F18.1)', 'Expected        = ', expected
  if (dot == expected) then
     print '(A,F10.6,A)', 'Verification: PASS   (time ', t1 - t0, ' s)'
  else
     print '(A,F10.6,A)', 'Verification: FAIL   (time ', t1 - t0, ' s)'
  end if

  !$omp target exit data map(delete: a, b)
  deallocate(a, b)
end program dot_gpu_only
