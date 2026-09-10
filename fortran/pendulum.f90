program pendulum
  implicit none
  integer, parameter  :: dp=kind(0.d0)
  real(dp), parameter :: y0(4) = [2.0_dp, 2.0_dp, 0.0_dp, -1.0_dp]
  real(dp), parameter :: h = 0.2_dp
  real(dp), parameter :: T = 10000
  integer, parameter  :: nsteps = ceiling(T / h)

  integer :: iter
  real(dp) :: y(4,nsteps+1)
  real(dp) :: start, finish

  do iter = 1,10
     call cpu_time(start)
     call runge5(y0, h, nsteps, y)
     call cpu_time(finish)
     print '("Time = ",f9.6," seconds.")', finish-start
  end do
end program pendulum

subroutine runge5(y0, h, nsteps, y)
  implicit none
  integer, parameter  :: dp=kind(0.d0)
  real(dp), intent(in) :: y0(4)
  real(dp), intent(in) :: h
  integer, intent(in)  :: nsteps
  real(dp), intent(out) :: y(4,nsteps+1)
  real(dp) :: k1(4), k2(4), k3(4), k4(4), k5(4), k6(4), yn(4)
  integer :: n
  
  y(:,1) = y0
  do n = 1,nsteps
     yn = y(:,n)
     call fpend(yn, k1)
     call fpend(yn + h*k1/5, k2)
     call fpend(yn + h*2*k2/5, k3)
     call fpend(yn + h*9*k1/4 - h*5*k2 + h*15*k3/4, k4)
     call fpend(yn - h*63*k1/100 + h*9*k2/5 - h*13*k3/20 + h*2*k4/25, k5)
     call fpend(yn - h*6*k1/25 + h*4*k2/5 + h*2*k3/15 + h*8*k4/75, k6)
     y(:,n+1) = yn + h*(17*k1 + 100*k3 + 2*k4 - 50*k5 + 75*k6) / 144
  end do
end subroutine runge5

subroutine fpend(y,f)
  implicit none
  integer, parameter  :: dp=kind(0.d0)
  real(dp), intent(in) :: y(4)
  real(dp), intent(out) :: f(4)
  real(dp) :: th1, th2, om1, om2, th1dot, th2dot, om1dot, om2dot
  real(dp) :: s1, c1, s2, c2, sd, cd, s12, denom
  
  th1 = y(1)
  th2 = y(2)
  om1 = y(3)
  om2 = y(4)
  
  s1 = sin(th1)          ! each sin/cos pair costs one sincos call
  c1 = cos(th1)
  s2 = sin(th2)
  c2 = cos(th2)
  sd = s1*c2 - c1*s2     ! sin(th1-th2)
  cd = c1*c2 + s1*s2     ! cos(th1-th2)
  s12 = sd*c2 - cd*s2    ! sin(th1-2*th2)
  denom = 2 + 2*sd**2    ! 3-cos(2*th1-2*th2)
  
  th1dot = om1
  th2dot = om2
  om1dot = (-3*s1 - s12 - 2*sd*(om2**2 + om1**2*cd)) / denom
  om2dot = 2*sd*(2*om1**2 + 2*c1 + om2**2*cd) / denom

  f = [th1dot, th2dot, om1dot, om2dot]
end subroutine fpend

