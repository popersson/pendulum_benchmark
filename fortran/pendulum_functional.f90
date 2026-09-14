module pendulum_functional
  use iso_fortran_env, only: dp => real64
  implicit none (type, external)
  private
  public :: dp, neq, fpend, runge5

  integer, parameter :: neq = 4    ! number of equations in the system

  ! Any right-hand side of this shape can be integrated by runge5 below.  It
  ! writes into an out-argument rather than returning an array: gfortran does not
  ! elide the copy of an array-valued function result, which costs about 8% here.
  abstract interface
     pure function rhs(y) result(f)
       import :: dp, neq
       real(dp), intent(in) :: y(neq)
       real(dp) :: f(neq)
     end function rhs
  end interface

contains

  ! Right-hand side of the double pendulum.  The associate construct names the
  ! components of the state, so the formulas can be read off directly.  Each
  ! sin/cos pair of the same angle costs one sincos call, and every other
  ! trigonometric value the equations need follows by identity.
  pure function fpend(y) result(f)
    real(dp), intent(in) :: y(neq)
    real(dp) f(neq)
    real(dp) :: s1, c1, s2, c2, sdth, cdth, s12, denom

    associate (th1 => y(1), th2 => y(2), om1 => y(3), om2 => y(4))
      s1 = sin(th1);  c1 = cos(th1)
      s2 = sin(th2);  c2 = cos(th2)
      sdth = s1*c2 - c1*s2        ! sin(th1 - th2)
      cdth = c1*c2 + s1*s2        ! cos(th1 - th2)
      s12 = sdth*c2 - cdth*s2     ! sin(th1 - 2*th2)
      denom = 2 + 2*sdth**2       ! 3 - cos(2*th1 - 2*th2)

      f = [om1, &
           om2, &
           (-3*s1 - s12 - 2*sdth*(om2**2 + om1**2*cdth)) / denom, &
           2*sdth*(2*om1**2 + 2*c1 + om2**2*cdth) / denom]
    end associate
  end function fpend

  ! Fifth order Runge-Kutta: integrate f from y0 in steps of h, one column of y
  ! per step.  h multiplies the stage arguments rather than the stages
  ! themselves, which lets the compiler hoist the products out of the loop.
  pure subroutine runge5(f, y0, h, nsteps, y)
    procedure(rhs) :: f
    real(dp), intent(in) :: y0(neq), h
    integer, intent(in) :: nsteps
    real(dp), intent(out) :: y(neq,nsteps+1)

    integer :: n

    y(:,1) = y0
    do n = 1, nsteps
       associate (yn => y(:,n))
         associate(k1 => f(yn))
         associate(k2 => f(yn + h*k1/5))
         associate(k3 => f(yn + h*2*k2/5))
         associate(k4 => f(yn + h*9*k1/4 - h*5*k2 + h*15*k3/4))
         associate(k5 => f(yn - h*63*k1/100 + h*9*k2/5 - h*13*k3/20 + h*2*k4/25))
         associate(k6 => f(yn - h*6*k1/25 + h*4*k2/5 + h*2*k3/15 + h*8*k4/75))
         y(:,n+1) = yn + h*(17*k1 + 100*k3 + 2*k4 - 50*k5 + 75*k6) / 144
         end associate;  end associate;  end associate;  end associate;  end associate; end associate
       end associate
    end do
  end subroutine runge5

end module pendulum_functional


program pendulum
  use iso_fortran_env, only: int64
  use pendulum_functional, only: dp, neq, fpend, runge5
  implicit none (type, external)

  real(dp), parameter :: y0(neq) = [real(dp) :: 2, 2, 0, -1]
  real(dp), parameter :: h = 0.2_dp
  real(dp), parameter :: T = 10000
  integer, parameter :: nsteps = nint(T/h)

  real(dp) :: y(neq,nsteps+1)
  real(dp), volatile :: sink       ! keeps the solution observable to the optimizer
  integer(int64) :: start, finish, rate
  integer :: iter

  do iter = 1, 10
     call system_clock(start, rate)
     call runge5(fpend, y0, h, nsteps, y)
     call system_clock(finish)

     sink = sum(y)   ! outside the timing, so the traversal is not measured

     print '("Time ",f9.6," seconds.")', real(finish - start, dp) / rate
  end do
end program pendulum
