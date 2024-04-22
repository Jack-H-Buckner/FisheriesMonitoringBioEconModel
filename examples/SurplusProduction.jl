module SurplusProduction

using Distributions 
using Roots

# function convert_harvest(B, H)
#     Fmax = 0.9
#     Smin = -log(1 - Fmax)

    
#     if B > H
#         s = [-log((B-H)/B), Smin]
#         S = s[argmin(s)]
#     else
#         S = Smin
#     end 
#     H = B*(1-exp(-S))
#     return H, B*exp(-S)
# end 

function convert_harvest(B, H)
    Fmax = 0.9
    Smin = -log(1 - Fmax)

    if B > H
        if -log((B-H)/B) < Smin
            S = -log((B-H)/B)
        else
            S = Smin
        end
    else
        S = Smin
    end 
    H = B*(1-exp(-S))
    return H, B*exp(-S)
end 

function convert_harvest_saturating_effort(biomass, TAC, max_effort, biomass_half_max_effort)
    b = 1/biomass_half_max_effort
    effort = b*max_effort*biomass/(1+b*biomass)
    harvest = effort*biomass
    harvest = min(harvest,TAC)
    return harvest, biomass-harvest
end 
function convert_harvest2(B, H)
    Fmax = 0.9
    Smin = -log(1 - Fmax)

    b = 10
    if B > H
        s = -log((B-H)/B)
        S = -log(b,b^(Smin-s)+1)+Smin
    else
        S = Smin
    end 
    H = B*(1-exp(-S))
    return H, B*exp(-S)
end 

function p_over_fish(H,state,pars)
    Bmin,Bcrt,Fmin,Fmax = pars
    xhat,cov = state
    a = Fmax/(Bcrt-Bmin)
    b = - Bmin*Fmax/(Bcrt-Bmin)
    Bstar = H/Fmax
    
    if Bstar < Bcrt
        Bstar = (-b + sqrt(b^2+4*a*H))/(2*a)
    end
    
    if H/Bstar < Fmin
        Bstar = H/Fmin
    end 
    
    return Distributions.cdf(Distributions.Normal(xhat[1], sqrt(cov[1,1])),log(Bstar))    
end 


function pstar_sigma(pstar,state,pars,Hmax)
    Roots.find_zero(H->p_over_fish(H,state,pars)-pstar,[0.0,500]) 
end 



function p_escapement(H,state,pars)
    target = pars[1]
    xhat,cov = state

    Bstar = target + H

    return Distributions.cdf(Distributions.Normal(xhat[1], sqrt(cov[1,1])),log(Bstar))    
end 


function pstar_escapement(pstar,state,pars,Hmax)
    if p_escapement(0,state,pars) > pstar
        return 0.0
    end 
    Roots.find_zero(H->p_escapement(H,state,pars)-pstar,[0,Hmax]) 
end 



pars = [10.0, 100]
function Bt(x,actions,aux,pars)
    Ht,obs = actions
    B = exp(x[1])
    #epsilon = exp(x[2]) # production shocks

    H,B=convert_harvest(B, Ht)

    Bprime = (1+ 4*pars[1]/pars[2])*B - 4*pars[1]*B^2/pars[2]^2
    #Bprime *= epsilon
    if Bprime <0
        Bprime = 1.0
    end 
    return [log(Bprime)]
end 


function Bt_allee(x,actions,aux,MSY_,K)
    Ht,obs = actions
    B = exp(x[1])

    H,B=convert_harvest(B, Ht)
    
    if B < K/2
        K *= 3/4
        Bprime = B+27*MSY_/(4*K^2)*B^2-27*MSY_/(4*K^3)*B^3 
    else
        Bprime = (1+ 4*MSY_/K)*B - 4*MSY_*B^2/K^2
    end
    
    if Bprime <0
        Bprime = 1.0
    end
    return [log(Bprime)]
end 


function Bt_pstar(x,actions,aux,pars)
    obs = actions
    Ht = aux
    B = exp(x[1])
    #epsilon = exp(x[2]) # production shocks

    H,B=convert_harvest(B, Ht)

    Bprime = (1+ 4*pars[1]/pars[2])*B - 4*pars[1]*B^2/pars[2]^2
    #Bprime *= epsilon
    if Bprime <0
        Bprime = 1.0
    end 
    return [log(Bprime)]
end 


T(x,Ht) = Bt(x,Ht,pars) 
T!(x,Ht) = Bt!(x,Ht,pars) 


function Bt_2(x,actions,aux,pars)
    Ht,obs = actions
    B = exp(x[1])
    epsilon = exp(x[2]) # production shocks

    H,B=convert_harvest(B, Ht)

    Bprime = (1+ 4*pars[1]/pars[2]*epsilon )*B - 4*pars[1]*B^2/pars[2]^2

    if Bprime <0
        Bprime = 1.0
    end 
    return [log(Bprime),0.0]
end 

function Bt_2!(x,actions,aux,pars) 
    Ht,obs = actions
    B = exp(x[1])
    epsilon = exp(x[2]) # production shocks

    H,B=convert_harvest(B, Ht)
    
    Bprime = (1+ 4*pars[1]/pars[2]*epsilon)*B - 4*pars[1]*B^2/pars[2]^2

    if Bprime <0
        Bprime = 1.0
    end 
    x[1] = log(Bprime)
end 



function R_obs(x,Ht)
    Fmax = -log(0.9)
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    return H 
end

function R(x,actions,aux,c1,c2,c3)
    Ht,Ot =actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    return H-c1*H/B-c2*H^2-c3*Ot
end 


function R_nonuse(x,actions,aux,price,c1,c2,c3,b,maxV)
    Ht,Ot =actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    return price*H-c1*H/B-c2*H^2-c3*Ot + b*maxV*B/(1+b*B)
end 

function R_alometric(x,actions,aux,c,p,c3)
    Ht,Ot =actions
    B = exp(x[1])
    H1,B1=convert_harvest(B, Ht)

    return c*(H1/c)^p - (Ot-1)*c3
end

function R_log(x,actions,aux,c1,c2,c3)
    Ht,Ot =actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    return log(H.+0.5)-c1*H/B-c2*H^2-c3*Ot
end 

function Rpstar(x,actions,aux,c1,c2,c3)
    Ht = aux
    Ot = actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    
    return H-c1*H/B-c2*H^2-c3*(Ot-1)
end 

function Rpstar_nonuse(x,actions,aux,price,c1,c2,c3,b,maxV)
    Ht = aux
    Ot = actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    
    return price*H-c1*price*H/B-c2*price*H^2-c3*Ot + b*maxV*B/(1+b*B)
end


function Rpstar_nonuse_saturating_effort(x,actions,aux,price,c1,c2,c3,b,maxV, max_effort, biomass_half_max_effort)
    Ht = aux
    Ot = actions
    B = exp(x[1])
    H,B=convert_harvest_saturating_effort(B, Ht, max_effort, biomass_half_max_effort)

    return price*H-c1*price*H/B-c2*price*H^2-c3*Ot + b*maxV*B/(1+b*B)
end

function Rpstar_penelty(x,actions,aux,threshold, penelty,c1,c2,c3)
    Ht = aux
    Ot = actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    v = 0
    if B < threshold
        v += -1*penelty
    end 
    v += H-c1*H/B-c2*H^2-c3*Ot
    return v
end 

function Rpstar_log(x,actions,aux,c1,c2,c3)
    Ht = aux
    Ot = actions
    B = exp(x[1])
    H,B=convert_harvest(B, Ht)
    return log(H.+0.1)-c1*H/B-c2*H^2-c3*Ot
end 


H = [1.0 0.0]
Sigma_N = zeros(2,2)
Sigma_N[1,1] = 0.001
Sigma_N[2,2] = 0.05


function Sigma_O(actions,aux)
    Ht,obs=actions
    Sigma_O = zeros(1,1)
    if obs == 3
        Sigma_O[1,1] = 0.01
    elseif obs == 2
        Sigma_O[1,1] = 0.05
    elseif obs == 1
        Sigma_O[1,1] = 0.05
    else
        Sigma_O[1,1] = 0.23
    end
    return Sigma_O
end 


function Sigma_O2(actions,aux)
    Ht,obs=actions
    Sigma_O = zeros(1,1)
    if obs == 3
        Sigma_O[1,1] = 0.01
    elseif obs == 2
        Sigma_O[1,1] = 0.05
    elseif obs == 1
        Sigma_O[1,1] = 0.25
    else
        Sigma_O[1,1] = 1.0
    end
    return Sigma_O
end 


## alternative functional form 

using NLsolve
Bmsy(r,k) = (-2*k + sqrt(4*k^2 + 4*(r-1)*k^2))/(2*(r-1))
msy(r,k) = r*Bmsy(r,k)/(1+Bmsy(r,k)*(r-1)/k)-Bmsy(r,k)
reparam_target(pars,MSY, ratio) = [msy(pars[1],pars[2])-MSY, msy(pars[1],pars[2])/Bmsy(pars[1],pars[2])-ratio]
function reparam(MSY,ratio)
    sol = NLsolve.nlsolve(x->reparam_target(x,MSY, ratio), [1.1,100.0])
    return sol.zero
end 


MSY = 10
ratio = 0.2
pars = reparam(MSY,ratio)

function Bt_alt(x,actions,aux,pars)
    Ht,obs = actions
    B = exp(x[1])
    #epsilon = exp(x[2]) # production shocks

    H,B=convert_harvest(B, Ht)

    Bprime = pars[1]*B/(1+(pars[1]-1)*B/pars[2]) 
    #Bprime *= epsilon
    if Bprime <=0.00001
        Bprime = 0.1
    end 
    return [log(Bprime)]
end 


function Bt_alee_alt(x,actions,aux,pars)
    Ht,obs = actions
    B = exp(x[1])
    K = pars[1]
    MSY_ = pars[2]
    
    H,B=convert_harvest(B, Ht)
    
    if B < K/2
        K *= 3/4
        Bprime = B+27*MSY_/(4*K^2)*B^2-27*MSY_/(4*K^3)*B^3 
    else
        Bprime = pars[3]*B/(1+(pars[3]-1)*B/pars[4]) 
    end
    
    if Bprime <=0.00001
        Bprime = 0.1
    end 
    return [log(Bprime)]
end 

function Bt_alt_pstar(x,actions,aux,pars)
    #obs = actions
    #Ht = aux
    #B = exp(x[1])
    #epsilon = exp(x[2]) # production shocks

    H,B=convert_harvest(exp(x[1]), aux)

    Bprime = pars[1]*B/(1+(pars[1]-1)*B/pars[2]) 
    #Bprime *= epsilon
    if Bprime <=0.000001
        Bprime += 0.1
    end 
    return [log(Bprime)+pars[3]]
end


function Bt_saturating_effort_pstar(x,actions,aux,pars,max_effort, biomass_half_max_effort)
    
    H,B=convert_harvest_saturating_effort(exp(x[1]), aux, max_effort, biomass_half_max_effort)

    Bprime = pars[1]*B/(1+(pars[1]-1)*B/pars[2]) 
    #Bprime *= epsilon
    if Bprime <=0.000001
        Bprime += 0.1
    end 
    return [log(Bprime)+pars[3]]
end




# define model and solver

include("../src/MDPsolver.jl")
mutable struct model
    params::AbstractVector{Float64}
    mod
    grid
    rewards
    Value
    VoI
    Policy
    Returns
end 




function init_model(MSY,Fmsy,pstar,SigmaN,sigma_a,sigma_p,c1,c2,c3,b,maxV,discount;price=1,N=50,CVmax=1.0,actions=[1,2])
    F_target = Fmsy
    B_threshold = MSY/Fmsy
    return init_model(MSY,Fmsy,F_target,B_threshold,pstar,SigmaN,sigma_a,sigma_p,c1,c2,c3,b,maxV,discount; 
                        price=price,N=N,CVmax=CVmax,actions=actions)
end 


function init_model(MSY,Fmsy,F_target,B_threshold,pstar,SigmaN,sigma_a,sigma_p,c1,c2,c3,b,maxV,discount;
                    price=1,N=50,CVmax=1.0,actions=[1,2])
    
    pars = [MSY,Fmsy,pstar,SigmaN,sigma_a,sigma_p,c1,c2,c3,discount]
    
    # Surplus production model 
    params = reparam(MSY,Fmsy);params = vcat(params,-0.5*SigmaN);T = (x,actions,aux) -> Bt_alt_pstar(x,actions,aux,params)
    SigmaN = [SigmaN;;]

    # Harvest model (pstar - sigma)
    Bmsy = MSY/Fmsy
    Fmin=0.02;Fmax=F_target;Bmin=0.25*B_threshold;Bcrit=B_threshold;Hmax=20*MSY
    aux = state -> pstar_sigma(pstar,state,(Bmin,Bcrit,Fmin,Fmax),Hmax)

    # new observation model 
    H = [1.0;;];sigmas = [sigma_p,sigma_a]#; actions = [1,2]
    SigmaO = (actions,aux) -> [sigmas[actions];;]

    # define model object to compute grid 
    mod=MDPsolver.BeliefStateTransitions.init_model(T,aux,H,actions,SigmaO,SigmaN,1)  

    # Define value function 
    Bmax = 4*MSY/Fmsy; Bmin = 0.01*MSY/Fmsy
    V = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)

    # value function and grid 
    grid=MDPsolver.BeliefStateTransitions.init_transitions("quadrature",mod,V;mQuad = 20)
    MDPsolver.BeliefStateTransitions.computeTransitions!(grid)
    
    # compute rewards 
    R = (x,actions,aux) -> SurplusProduction.Rpstar_nonuse(x,actions,aux,price,c1,c2,c3,b,maxV) # 
    rewards = MDPsolver.BeliefStateTransitions.init_rewards(R,mod,V)
    delta = 1/(1+discount)

    
    # solve value function 
    MDPsolver.BeliefStateTransitions.computeRewards!(rewards)
    MDPsolver.solve_parallel(grid,V,rewards,delta;threashold=10^-3,verbos=false)

    # Policy function 
    P = MDPsolver.ValueFunctions.init_Norm2DGrid_obs_policy(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    MDPsolver.policy_parallel!(P,grid,V,rewards,delta)
    
    # VoI
    values_1 = zeros(N,N);values_0 = zeros(N,N)
    for i in 1:N
        for j in 1:N
            values_1[i,j]=MDPsolver.bellman!(zeros(2),2,i,j,grid,V,rewards,delta).+ c3
            values_0[i,j]=MDPsolver.bellman!(zeros(2),1,i,j,grid,V,rewards,delta)
            vals = [values_1[i,j],values_0[i,j]]
        end 
    end 
    VoI = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    VoI.values = reshape(values_1.-values_0, N^2);MDPsolver.ValueFunctions.update1!(VoI)
   

    return model(pars,mod,grid,rewards,V,VoI,P,R)
end 





function init_model(MSY,Fmsy,F_target,B_threshold,max_effort,biomass_half_max_effort,pstar,
                    SigmaN,sigma_a,sigma_p,c1,c2,c3,b,maxV,discount;
                    price=1,N=50,CVmax=1.0,actions=[1,2])
    print("here ")
    pars = [MSY,Fmsy,pstar,SigmaN,sigma_a,sigma_p,c1,c2,c3,discount]
    
    # Surplus production model 
    params = reparam(MSY,Fmsy);params = vcat(params,-0.5*SigmaN); 
    T = (x,actions,aux) -> Bt_saturating_effort_pstar(x,actions,aux,params,max_effort, biomass_half_max_effort)
    SigmaN = [SigmaN;;]

    # Harvest model (pstar - sigma)
    Bmsy = MSY/Fmsy
    Fmin=0.02;Fmax=F_target;Bmin=0.25*B_threshold;Bcrit=B_threshold;Hmax=20*MSY
    aux = state -> pstar_sigma(pstar,state,(Bmin,Bcrit,Fmin,Fmax),Hmax)

    # new observation model 
    H = [1.0;;];sigmas = [sigma_p,sigma_a]#; actions = [1,2]
    SigmaO = (actions,aux) -> [sigmas[actions];;]

    # define model object to compute grid 
    mod=MDPsolver.BeliefStateTransitions.init_model(T,aux,H,actions,SigmaO,SigmaN,1)  

    # Define value function 
    Bmax = 4*MSY/Fmsy; Bmin = 0.01*MSY/Fmsy
    V = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)

    # value function and grid 
    grid=MDPsolver.BeliefStateTransitions.init_transitions("quadrature",mod,V;mQuad = 20)
    MDPsolver.BeliefStateTransitions.computeTransitions!(grid)
    
    # compute rewards 
    R = (x,actions,aux) -> SurplusProduction.Rpstar_nonuse_saturating_effort(x,actions,aux,price,c1,c2,c3,b,maxV,
                                                                             max_effort,biomass_half_max_effort) # 
    rewards = MDPsolver.BeliefStateTransitions.init_rewards(R,mod,V)
    delta = 1/(1+discount)

    
    # solve value function 
    MDPsolver.BeliefStateTransitions.computeRewards!(rewards)
    MDPsolver.solve_parallel(grid,V,rewards,delta;threashold=10^-3,verbos=false)

    # Policy function 
    P = MDPsolver.ValueFunctions.init_Norm2DGrid_obs_policy(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    MDPsolver.policy_parallel!(P,grid,V,rewards,delta)
    
    # VoI
    values_1 = zeros(N,N);values_0 = zeros(N,N)
    for i in 1:N
        for j in 1:N
            values_1[i,j]=MDPsolver.bellman!(zeros(2),2,i,j,grid,V,rewards,delta).+ c3
            values_0[i,j]=MDPsolver.bellman!(zeros(2),1,i,j,grid,V,rewards,delta)
            vals = [values_1[i,j],values_0[i,j]]
        end 
    end 
    VoI = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    VoI.values = reshape(values_1.-values_0, N^2);MDPsolver.ValueFunctions.update1!(VoI)
   

    return model(pars,mod,grid,rewards,V,VoI,P,R)
end 

mutable struct model2
    params::AbstractVector{Float64}
    mod
    grid
    rewards
    Value
    VoI
    Policy
    Returns
end 


function init_model2(MSY,Fmsy,SigmaN,sigma_a,sigma_p,p,c3,discount;N=50,CVmax=1.0,step=2.5)
    
    pars = [MSY,Fmsy,SigmaN,sigma_a,sigma_p,p,c3,discount]
    
    # Surplus production model 
    params = reparam(MSY,Fmsy);params = vcat(params,-0.5*SigmaN); T = (x,actions,aux) -> Bt_alt(x,actions,aux,params)
    SigmaN = [SigmaN;;]
    aux = state -> 1
    
    # new observation model 
    H = [1.0;;];sigmas = [sigma_p,sigma_a]#; actions = [1,2]
    SigmaO = (action,aux) -> [sigmas[round(Int,action[2])];;]
    
    harvest = collect(0.0:step:50)
    observations = [1,2]
    actions = reshape(collect(Iterators.product(harvest, observations)), length(harvest)*length(observations))


    # define model object to compute grid 
    mod=MDPsolver.BeliefStateTransitions.init_model(T,aux,H,actions,SigmaO,SigmaN,1)  

    # Define value function 
    Bmax = 2.0*MSY/Fmsy; Bmin = 0.1*MSY/Fmsy
    println(log(CVmax^2+1))
    V = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)

    # value function and grid 
    grid=MDPsolver.BeliefStateTransitions.init_transitions("quadrature",mod,V;mQuad = 20)
    MDPsolver.BeliefStateTransitions.computeTransitions!(grid)
    
    # compute rewards 
    R = (x,actions,aux) -> R_alometric(x,actions,aux,MSY,p,c3) # 
    rewards = MDPsolver.BeliefStateTransitions.init_rewards(R,mod,V)
    delta = 1/(1+discount)

    
    # solve value function 
    MDPsolver.BeliefStateTransitions.computeRewards!(rewards)
    MDPsolver.solve_parallel(grid,V,rewards,delta;threashold=10^-3,verbos=false)
    
    # Policy function 
    P = MDPsolver.ValueFunctions.init_Norm2DGrid_policy(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    MDPsolver.policy_parallel!(P,grid,V,rewards,delta)
    
    # VoI
    values_1 = zeros(N,N);values_0 = zeros(N,N)
    for i in 1:N
        for j in 1:N
            values_1[i,j]=MDPsolver.bellman!(zeros(2),2,i,j,grid,V,rewards,delta).+ c3
            values_0[i,j]=MDPsolver.bellman!(zeros(2),1,i,j,grid,V,rewards,delta)
            vals = [values_1[i,j],values_0[i,j]]
        end 
    end 
    VoI = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    VoI.values = reshape(values_1.-values_0, N^2);MDPsolver.ValueFunctions.update1!(VoI)
   
    
    return model(pars,mod,grid,rewards,V,VoI,P,R)
end 




mutable struct model3
    params::AbstractVector{Float64}
    mod
    grid
    rewards
    Value
    VoI
    Policy
    Returns
end 


function init_model3(MSY,Fmsy,SigmaN,sigma_a,sigma_p,c1,c2,c3,low_harvest_penelty,b,maxV,discount;
                        N=50,CVmax=1.0,pi_MSY=10.0,
                        harvest = vcat(vcat(collect(0.0:1.0:30), 
                                        collect(32:2.0:60)), 
                                        collect(64:4.0:100)),
                        observations = [1,2])
    
    pars = [MSY,Fmsy,SigmaN,sigma_a,sigma_p,c1,c2,c3,b,maxV,discount]
    
    # Surplus production model 
    params = reparam(MSY,Fmsy);params = vcat(params,-0.5*SigmaN); T = (x,actions,aux) -> Bt_alt(x,actions,aux,params)
    SigmaN = [SigmaN;;]
    aux = state -> 1
    
    # new observation model 
    H = [1.0;;];sigmas = [sigma_p,sigma_a]#; actions = [1,2]
    SigmaO = (action,aux) -> [sigmas[round(Int,action[2])];;]
    
    #observations = [1,2]
    actions = reshape(collect(Iterators.product(harvest, observations)), length(harvest)*length(observations))


    # define model object to compute grid 
    mod=MDPsolver.BeliefStateTransitions.init_model(T,aux,H,actions,SigmaO,SigmaN,1)  

    # Define value function 
    Bmax = 4.0*MSY/Fmsy; Bmin = 0.01*MSY/Fmsy
    V = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)

    # value function and grid 
    grid=MDPsolver.BeliefStateTransitions.init_transitions("quadrature",mod,V;mQuad = 20)
    MDPsolver.BeliefStateTransitions.computeTransitions!(grid)
    
    # compute rewards 
    price = (pi_MSY + c1*Fmsy + c2*MSY^2)/MSY
    function R(x,actions,aux) 
        if actions[1] > 5.0 
            return R_nonuse(x,actions,aux,price,c1,c2,c3,b,maxV)  + c3
        else
            return R_nonuse(x,actions,aux,price,c1,c2,c3,b,maxV) - low_harvest_penelty + c3
        end
    end 
    rewards = MDPsolver.BeliefStateTransitions.init_rewards(R,mod,V)
    delta = 1/(1+discount)

    
    # solve value function 
    MDPsolver.BeliefStateTransitions.computeRewards!(rewards)
    MDPsolver.solve_parallel(grid,V,rewards,delta;threashold=10^-5,verbos=false)
    
    # Policy function 
    P = MDPsolver.ValueFunctions.init_Norm2DGrid_policy(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    MDPsolver.policy_parallel!(P,grid,V,rewards,delta)
    
    # VoI
    values_1 = zeros(N,N);values_0 = zeros(N,N)
    for i in 1:N
        for j in 1:N
            values_1[i,j]=MDPsolver.bellman!(zeros(2),2,i,j,grid,V,rewards,delta).+ c3
            values_0[i,j]=MDPsolver.bellman!(zeros(2),1,i,j,grid,V,rewards,delta)
            vals = [values_1[i,j],values_0[i,j]]
        end 
    end 
    VoI = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    VoI.values = reshape(values_1.-values_0, N^2);MDPsolver.ValueFunctions.update1!(VoI)
   
    
    return model(pars,mod,grid,rewards,V,VoI,P,R)
end 






function compute_p(model,threshold;T=10000,Nmc=1000)
    MSY=model.params[1];Fmsy=model.params[2]
    x0 = [log(MSY/Fmsy)];s0 = ([log(MSY/Fmsy)], [0.25;;])
    filter = MDPsolver.BeliefStateTransitions.ParticleFilters.init(Nmc,Distributions.MvNormal(s0[1],s0[2]))
    dat = MDPsolver.BeliefStateTransitions.simulation(x0,s0,T,filter,model.Policy,model.mod,model.Returns)
    B = broadcast(i -> exp(dat[1][i][1]), 100:T)
    return sum(B.<threshold)/length(B)
end 


function MeanV(model;T=10000,Nmc=1000)
    MSY=model.params[1];Fmsy=model.params[2]
    x0 = [log(MSY/Fmsy)];s0 = ([log(MSY/Fmsy)], [0.25;;])
    filter = MDPsolver.BeliefStateTransitions.ParticleFilters.init(Nmc,Distributions.MvNormal(s0[1],s0[2]))
    
    dat = MDPsolver.BeliefStateTransitions.simulation(x0,s0,T,filter,model.Policy,model.mod,model.Returns)

    R = broadcast(i -> dat[5][i][1][1], 1:T)
    V = 0; EV = 0
    delta = 1/(1+model.params[end])
    for i in 0:(T-1)
        V *= delta
        V += R[end-i]
        if i > 150
            EV += V/(T-150) 
        end 
    end 

    return EV
end 




end 