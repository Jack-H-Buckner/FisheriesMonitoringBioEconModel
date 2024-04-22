module BaseParams
using Sobol

# base vlaue
Fmsy = 0.2
pstar = 0.4
tau = 0.05
sigma_a = 0.2
sigma_p = 2.0
H_weight = 10
NCV_weight = 10
c1 = 0.1
c2 = 0.05
b = 1.5
discount = 0.05
MSY = 10.0
monitoring_costs = 1.0
buffer = 0.1
price = 7.5
c = 50.0

# lower bound
lFmsy = 0.1
lpstar = 0.3
ltau = 0.025
lsigma_a = 0.1
lsigma_p = 1.0
lH_weight = 1.0
lNCV_weight = 1.0
lc1 = 0.0
lc2 = 0.0
lb = 1.2
ldiscount = 0.01
lbuffer = 0.0

# upper bound 
uFmsy = 0.3
upstar = 0.5
utau = 0.1
usigma_a = 0.5
usigma_p = 3.0
uH_weight = 15.0
uNCV_weight = 15.0
uc1 = 0.5
uc2 = 0.1
ub = 2.0
udiscount = 0.1
ubuffer = 0.3

function scale_sample(x,u,l)
    return l + (u-l)*x
end

function sample(s)
    
    x = next!(s)
    Fmsy = scale_sample(x[1],uFmsy,lFmsy)
    pstar = scale_sample(x[2],upstar,lpstar)
    tau = scale_sample(x[3],utau,ltau)
    sigma_a = scale_sample(x[4],usigma_a,lsigma_a)
    sigma_p = scale_sample(x[5],usigma_p,lsigma_p)
    H_weight = scale_sample(x[6],uH_weight,lH_weight)
    NCV_weight = scale_sample(x[7],uNCV_weight,lNCV_weight)
    c1 = scale_sample(x[8],uc1,lc1)
    c2 = scale_sample(x[9],uc2,lc2)
    b = scale_sample(x[10],ub,lb)
    discount = scale_sample(x[11],udiscount,ldiscount)

    return (Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount)
end 


function sample_simpler(s)
    
    x = next!(s)
    Fmsy = scale_sample(x[1],uFmsy,lFmsy)
    buffer = scale_sample(x[2],ubuffer,lbuffer)
    tau = scale_sample(x[3],utau,ltau)
    sigma_a = scale_sample(x[4],usigma_a,lsigma_a)
    sigma_p = scale_sample(x[5],usigma_p,lsigma_p)
    H_weight = scale_sample(x[6],uH_weight,lH_weight)
    NCV_weight = scale_sample(x[7],uNCV_weight,lNCV_weight)
    c1 = scale_sample(x[8],uc1,lc1)
    c2 = scale_sample(x[9],uc2,lc2)
    b = scale_sample(x[10],ub,lb)
    discount = scale_sample(x[11],udiscount,ldiscount)

    return (Fmsy,buffer,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount)
end 


function sample_profit_feedback(s)
    
    x = next!(s)
    Fmsy = scale_sample(x[1],uFmsy,lFmsy)
    buffer = scale_sample(x[2],ubuffer,lbuffer)
    tau = scale_sample(x[3],utau,ltau)
    sigma_a = scale_sample(x[4],usigma_a,lsigma_a)
    sigma_p = scale_sample(x[5],usigma_p,lsigma_p)
    price = scale_sample(x[6],uH_weight,lH_weight)
    NCV_weight = scale_sample(x[7],uNCV_weight,lNCV_weight)
    b = scale_sample(x[8],ub,lb)
    discount = scale_sample(x[9],udiscount,ldiscount)

    return (Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount)
end 


end # module 