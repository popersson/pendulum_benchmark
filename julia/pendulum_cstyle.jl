function runge5(f, y0, h, N)
    y = zeros(length(y0), N+1)
    y[:,1] = y0
    tmp,k1,k2,k3,k4,k5,k6 = zeros(4),zeros(4),zeros(4),zeros(4),zeros(4),zeros(4),zeros(4)

    for n = 1:N
        for i = 1:4; tmp[i] = y[i,n]; end
        f(tmp, k1)
        
        for i = 1:4; tmp[i] = y[i,n] + h*k1[i]/5; end
        f(tmp, k2)
        
        for i = 1:4; tmp[i] = y[i,n] + 2h*k2[i]/5; end
        f(tmp, k3)
        
        for i = 1:4; tmp[i] = y[i,n] + 9h*k1[i]/4 - 5h*k2[i] + 15h*k3[i]/4; end
        f(tmp, k4)
        
        for i = 1:4; tmp[i] = y[i,n] - 63h*k1[i]/100 + 9h*k2[i]/5 - 13h*k3[i]/20 + 2h*k4[i]/25; end
        f(tmp, k5)
        
        for i = 1:4; tmp[i] = y[i,n] - 6h*k1[i]/25 + 4h*k2[i]/5 + 2h*k3[i]/15 + 8h*k4[i]/75; end
        f(tmp, k6)
        
        for i = 1:4; y[i,n+1] = y[i,n] + h*(17*k1[i] + 100*k3[i] + 2*k4[i] - 50*k5[i] + 75*k6[i]) / 144; end
    end
    return y
end

function fpend(y,f)
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
    f[1],f[2],f[3],f[4] = θ1dot,θ2dot,ω1dot,ω2dot
end

y0 = [2,2,0,-1]
h = 0.2
T = 10000
N = round(Int, T/h)

for iter = 1:10
    @time y = runge5(fpend, y0, h, round(Int, T/h));
end

#@btime runge5(fpend, y0, h, N)
