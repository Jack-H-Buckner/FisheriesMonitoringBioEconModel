"""
A size structured popualtion modelw thi two state variables

This is an updated version of the model in the BiomassDynaics.jl file
That tracks the biomass and average size rather than numbers and biomass
"""
module BiomassDynamics2
using KalmanFilters

using NLsolve
using Optim

##### excess production and equilibrium analysis ######

function excess_production(x,pars)
    k, winfty,wr,m,r,b,sigma_R,Fmax= pars
    N = x[1]; B = x[2]
    return wr*r*B/(1 + b*B) + (1-m)*(N*winfty*(1-k)+k*B) - B 
end 


function sol_equib(x,F,pars)
    k, winfty,wr,m,r,b,sigma_R,Fmax = pars
    N = x[1]; B = x[2]
    y1 = exp(-F)*(1-m)*N + r*B/(1 + b*B) -N 
    y2 = wr*r*B/(1 + b*B) + exp(-F)*(1-m)*(N*winfty*(1-k)+k*B)- B 
    return [y1,y2]
end 

function equib(F, pars)
    x0 = [200.0,100]
    sol = nlsolve(x -> sol_equib(x,F,pars), x0)
    B = sol.zero[2]; Wbar = sol.zero[2]/sol.zero[1]
    return [B,Wbar]
end 


function sol_Fmsy(F,pars)
    eq = equib(F, pars)
    return eq[2]*(1-exp(-F))
end 


function Fmsy(pars)
    sol = Optim.optimize(F -> -1*sol_Fmsy(F,pars), 0.00001, 0.2)
    return -1*sol.minimum, sol.minimizer
end 


function FtoH_eq(F,pars)
    x = equib(F, pars)
    return x[2]*(1-exp(-F))
end 


function MSY(pars)
    F = Fmsy(pars)
    return FtoH_eq(F[2],pars)
end 



##### Population dynamic functions ######


function deterministic(x,Ht,pars)
    # unpack parameters
    k, winfty,wr,m,r,b,sigma_R,Fmax = pars
    
    # unpack states
    B=exp(x[1]);Wbar=exp( x[2] );v=x[3]
    N=B/Wbar
    F = Fmax
    if B > Ht  
        s = [Fmax,-log((B-Ht)/B)]
        F = s[argmin(s)]
    else
        F = Fmax
    end
     
    Nt = exp(-F)*(1-m)*N + r*B*exp(v)/(1+b*B)
        
    Bt = exp(-F)*(1-m)*(N*winfty*(1-k)+k*B) + wr*r*B*exp(v-0.5*sigma_R^2)/(1+b*B)
    
    Wbar=Bt/Nt

    return [log(Bt),log(Wbar),0], (1-exp(-F))*(1-m)*(N*winfty*(1-k)+k*B)
    
end 


function deterministic!(x,Ht,pars)
    # unpack parameters
    k, winfty,wr,m,r,b,sigma_R,Fmax = pars
    
    # unpack states
    B=exp(x[1]);Wbar=exp(x[2]);v=x[3]
    N=B/Wbar
    F = Fmax
    if B > Ht  
        s = [Fmax,-log((B-Ht)/B)]
        F = s[argmin(s)]
    else
        F = Fmax
    end
     
    Nt = exp(-F)*(1-m)*N + r*B*exp(v)/(1+b*B)
        
    Bt = exp(-F)*(1-m)*(N*winfty*(1-k)+k*B) + wr*r*B*exp(v-0.5*sigma_R^2)/(1+b*B)
    
    Wbar=Bt/Nt
    
    x[1] = log(Bt)
    
    x[2] = log(Wbar)
    
    x[3] = 0.0

    
end 



## transition function for simulations 
function stochastic(x, Ht,pars)
    # unpack parameters
    k, winfty,wr,m,r,b,sigma_R,Fmax = pars
    
    # unpack states
    B=exp(x[1]);Wbar=exp(x[2]);v=x[3]
    N=B/Wbar
    F = Fmax
    if B > Ht  
        s = [Fmax,-log((B-Ht)/B)]
        F = s[argmin(s)]
    else
        F = Fmax
    end
    
    Nt = exp(-F)*(1-m)*N + r*B*exp(v)/(1+b*B)
        
    Bt = exp(-F)*(1-m)*(N*winfty*(1-k)+k*B) + wr*r*B*exp(v-0.5*sigma_R^2)/(1+b*B)
    
    Wbar=Bt/Nt
        
    H = (1-exp(-F))*(1-m)*(N*winfty*(1-k)+k*B)
    
    epsilon = rand(Distributions.Normal(0, sigma_R))
        
    xt = [log(Bt),log(Wbar),epsilon]

    return xt , H, F
        
end 



#####  Kalman filter #####

using Distributions 
        
function observation(x, Sigma_O)

    H = [1.0 0.0 0.0 ; 0.0 1.0 0.0]
    
    mu = H * x 
    return rand(Distributions.MvNormal(mu,Sigma_O))   
end 





###### simulations ######

mutable struct state
    xt::AbstractArray{Float64}
    Ht::Float64
    Ft::Float64
    xhat::AbstractArray{Float64}
    Sigma::AbstractMatrix{Float64}
end 

"""
initializes the popuatlion at 
"""
function init(pars)
    k, winfty,wr,m,r,b,sigma_R,Fmax = pars
    xt = vcat(log.(equib(0.0, pars)),[0])
    Ht = 0.0
    Ft = 0.0
    xhat = xt
    Sigma = [1.0 0 0; 0 1.0 0; 0 0 sigma_R]
    return state(xt,Ht,Ft,xhat,Sigma)
end 

function p_star_sigma(state,Ftarget,pstar)
    sigma_B = state.Sigma[2,2]
    dB = Distributions.LogNormal(state.xhat[2],sqrt(sigma_B))
    q = quantile(dB, pstar)

    Ht = q*(1-exp(-Ftarget))
    return Ht
end


function time_step!(state,pars,HCR)
    # unpack pars
    k, winfty,wr,m,r,b,sigma_R,Fmax = pars

    # evaluate HCR
    Ht = HCR(state)
    # simualte time step 
    
    if Ht > exp(state.xt[2])
        state.Ft = Fmax 
    else
        state.Ft = -log(exp(state.xt[2]) - Ht) + state.xt[2] 
    end
    
    
    state.xt, state.Ht = stochastic(state.xt, Ht,pars)  
    #println(state.Ht)
    # time update
    Sigma_N = [10^-6 0 0; 0 10^-6 0; 0 0 sigma_R]
    uKF = KalmanFilters.time_update(state.xhat,state.Sigma, x -> deterministic(x, Ht, pars),  Sigma_N)
    state.xhat = get_state(uKF)
    state.Sigma = get_covariance(uKF)
    
end 

function observation!(state,pars,Sigma_O)
    
    yt = observation(state.xt, Sigma_O)

    # observation update
    H = [0.0 1.0 0.0 ; -1.0 1.0 0.0]
    uKF = KalmanFilters.measurement_update(state.xhat,state.Sigma,yt,H,Sigma_O)
    state.xhat = get_state(uKF)
    state.Sigma = get_covariance(uKF)
    
end 



function simulation(state,pars,omega,Sigma_O,HCR,T)
    Bt = zeros(T)
    Ht = zeros(T)
    Ft = zeros(T)
    Bhat = zeros(T)
    Sigma_Bt = zeros(T)
    for t in 1:T
        BiomassDynamics.time_step!(state,pars,HCR)
        if mod(t,omega) == 0
            BiomassDynamics.observation!(state,pars,Sigma_O)
        end 
        Bt[t] = exp.(state.xt[2])
        Ht[t] = state.Ht
        Ft[t] = state.Ft
        Bhat[t] = state.xhat[2]
        Sigma_Bt[t] = sqrt(state.Sigma[2])
    end 
    
    return Bt, Ht, Ft, Bhat, Sigma_Bt
    
end 

function mean_variance_Ht(state,pars,omega,Sigma_O,HCR,T)
    Bt, Ht, Ft, Sigma_Bt = simulation(state,pars,omega,Sigma_O,HCR,T)
    Tbegin = floor(Int, T/10)
    Hseries = Ht[Tbegin:end]
    return sum(Hseries)/length(Hseries), sqrt(var(Hseries))
        
end 

#################################################
#####                                       #####
#####    define some default parameters     #####
#####                                       #####
#################################################

### general parameters ###
sigmaR = 0.2
Fmax = 0.95

### Long life history ###
k_long = 0.92
winfty_long = 1.0
wr_long = 0.0
m_long = 0.075

r_long = 0.9
b_long = 0.017252


### short life histry ###
k_short = 0.80
winfty_short = 1.0
wr_short = 0.00
m_short = 0.215

r_short = 2.4
b_short = 0.02

### pars lists
pars_long = (k_long,winfty_long,wr_long,m_long,r_long,b_long,sigmaR,Fmax)
pars_short = (k_short,winfty_short,wr_short,m_short,r_short,b_short,sigmaR,Fmax)


### sigma_O ###

Sigma_O_1 = [0.1 0; 0 0.01]



end 