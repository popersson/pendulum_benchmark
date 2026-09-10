using StaticArrays

function runge5(f, y0, h, N)
    y = zeros(length(y0), N+1)

    y[:,1] .= y0
    for n = 1:N
        yn = @view y[:,n]
        k1 = f(yn)
        k2 = f(yn + h*k1/5)
        k3 = f(yn + 2h*k2/5)
        k4 = f(yn + 9h*k1/4 - 5h*k2 + 15h*k3/4)
        k5 = f(yn - 63h*k1/100 + 9h*k2/5 - 13h*k3/20 + 2h*k4/25)
        k6 = f(yn - 6h*k1/25 + 4h*k2/5 + 2h*k3/15 + 8h*k4/75)
        y[:,n+1] = yn + h*(17k1 + 100k3 + 2k4 - 50k5 + 75k6) / 144
    end
    
    return y
end

@inline function fpend(y)
    θ1,θ2,ω1,ω2 = y
    s1,c1 = sincos(θ1)
    s2,c2 = sincos(θ2)
    sΔ = s1*c2 - c1*s2          # sin(θ1-θ2)
    cΔ = c1*c2 + s1*s2          # cos(θ1-θ2)
    s12 = sΔ*c2 - cΔ*s2         # sin(θ1-2θ2)
    denom = 2 + 2sΔ^2           # 3-cos(2θ1-2θ2)
    θ1dot = ω1
    θ2dot = ω2
    ω1dot = (-3*s1 - s12 - 2*sΔ*(ω2^2 + ω1^2*cΔ))/denom
    ω2dot = 2*sΔ*(2*ω1^2 + 2*c1 + ω2^2*cΔ)/denom
    return SVector(θ1dot,θ2dot,ω1dot,ω2dot)
end

y0 = SA[2,2,0,-1]
h = 0.2
T = 10000
N = round(Int, T/h)

for iter = 1:10
    @time y = runge5(fpend, y0, h, N);
end

#@btime runge5(fpend, y0, h, N)
