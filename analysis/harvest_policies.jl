"""
The file contains code to solve for the optimal policies for monitoring and harvest 
over arange of parameter values. 

- Monitoring costs
- Accuracy of active monitoring

Base + 2 scenarios with higher cost and 2 with lower accuracy = 5 scanarios

- Discount rate
- Risk aversion
- Non-consumptive value

Base + NCV + higher RA + higher Discount rate = 4 parameter sets

Computing: 
- 3 sets of beleif state transitions 
- 12 sets of reward function + reward matrices 
- 3* 12 = 36 sets of VFI runs

Outputs: 
- model object 
    - Harvest policy function 
    - Monitoring policy
    - value function 

Simulations:
- 
    - Harvest - mean +var + bootstrap CI
    - Monitoring costs - mean 
    - Abundance - mean +var + bootstrap CI
    - CV beleif state - mean +var + bootstrap CI


Set reward functions to have equal profits when fishing at {MSY, Fmsy}

pi_msy = p*MSY +c1 * Fmsy + c2 * MSY^2  

p = (pi_msy - c1 * Fmsy + c2 * MSY^2)/MSY 
"""
module harvest_policies

include("../src/BeliefStateTransitions.jl")
include("../src/MDPsolver.jl")
include("../src/MvGaussHermite.jl")
include("SurplusProduction.jl")
#import Pkg; Pkg.add("DataFrames");Pkg.add("Statistics"); Pkg.add("JLD2")

function set_up_model(pars;N=100,harvest = collect(0.0:1.0:50))
    
    # unpack parameters
    MSY,Fmsy,SigmaN,sigma_a,sigma_p,c1,c2,c3,b,maxV,discount = pars
    
    # Surplus production model 
    params = SurplusProduction.reparam(MSY,Fmsy)
    params = vcat(params,-0.5*SigmaN[1,1])
    T = (x,actions,aux) -> SurplusProduction.Bt_alt(x,actions,aux,params)
    aux = state -> 1
    
    # new observation model 
    H = [1.0;;];sigmas = [sigma_p,sigma_a]
    SigmaO = (action,aux) -> [sigmas[round(Int,action[2])];;]
    
    # Define action set
    observations = [1,2]
    actions = reshape(collect(Iterators.product(harvest, observations)), length(harvest)*length(observations))
    
    # Define value function
    Bmax = 4.0*MSY/Fmsy; Bmin = 0.01*MSY/Fmsy;CVmax=1.0
    V = MDPsolver.ValueFunctions.init_Norm2DGrid(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    P = MDPsolver.ValueFunctions.init_Norm2DGrid_policy(log(Bmax),log(Bmin),log(CVmax^2+1),N)
    
    # define model object to compute grid 
    mod=MDPsolver.BeliefStateTransitions.init_model(T,aux,H,actions,SigmaO,SigmaN,1)  
    
    # compute rewards 
    price = (MSY + c1*Fmsy + c2*MSY^2)/MSY
    R = (x,actions,aux) -> SurplusProduction.R_nonuse(x,actions,aux,price,c1,c2,c3,b,maxV) 
    
    
    delta = 1/(1+discount)
    
    return R, V, P, mod, delta
    
end 


function compute_state_transitions(V,mod)
    # value function and grid 
    grid=MDPsolver.BeliefStateTransitions.init_transitions("quadrature",mod,V;mQuad = 20)
    MDPsolver.BeliefStateTransitions.computeTransitions!(grid)
    
    return grid
end 

function compute_rewards(R,V,mod)
    # value function and grid 
    rewards = MDPsolver.BeliefStateTransitions.init_rewards(R,mod,V)

    # solve value function 
    MDPsolver.BeliefStateTransitions.computeRewards!(rewards)
    
    return rewards
end 


function VFI(grid,rewards, V, delta)
    V_ = deepcopy(V)
    MDPsolver.solve_parallel(grid,V_,rewards,delta;threashold=10^-3,verbos=false)
    return V_
end 


function compute_Policy(grid,rewards, V_, P,delta)
    P_ = deepcopy(P)
    MDPsolver.policy_parallel!(P_,grid,V_,rewards,delta)
    return P_
end 

using DataFrames
using Statistics
function simulation(mod,P,R_,pars;T=500,NMC=250,burnin=50)
    MSY=pars[1];Fmsy=pars[2];sigma0=0.2
    M = []
    H = []
    R = []
    mods = broadcast(i->deepcopy(mod),1:Threads.nthreads())
    T+=burnin
    Threads.@threads for i in 1:NMC 

        x0 = [log(2*MSY/Fmsy)];s0 = ([log(2*MSY/Fmsy)], [sigma0;;])
        quad = BeliefStateTransitions.MvGaussHermite.init_mutable(10,s0[1],s0[2])
        dat=BeliefStateTransitions.simulation_kf(x0,s0,T,P,mods[Threads.threadid()],R_,quad)
   
        
        M = vcat(M, broadcast(i -> dat[3][i][2], burnin:T))
        H = vcat(H, broadcast(i -> dat[3][i][1], burnin:T))
        R = vcat(R, broadcast(i -> dat[5][i][1], burnin:T))
            
    end 


    mean_M = sum(M)/length(M) .-1
    mean_H = sum(H)/length(H)
    mean_R = sum(R)/length(R)

    mean_Hbs = broadcast(i->sum( H[rand(1:length(H),length(H))] )/length(H),1:5000)
    mean_Rbs = broadcast(i->sum( R[rand(1:length(R),length(R))] )/length(R),1:5000)

    qH=quantile(mean_Hbs,[0.05,0.25,0.5,0.75,0.95])
    qR=quantile(mean_Rbs,[0.05,0.25,0.5,0.75,0.95])
    
    results = DataFrame(M=mean_M,H=qH[3],H005=qH[1],H025=qH[2],H075=qH[4],H095=qH[5],
                                 R=qR[3],R005=qR[1],R025=qR[2],R075=qR[4],R095=qR[5])

    results = DataFrame(variable = ["M","H","H","H","H","H","R","R","R","R","R"],
                        quantile = [0.5,0.5,0.05,0.25,0.75,0.95,0.5,0.05,0.25,0.75,0.95],
                        value = [mean_M,qH[3],qH[1],qH[2],qH[4],qH[5],qR[3],qR[1],qR[2],qR[4],qR[5]])


    return results 
end 


struct solution
    pars
    P
    V
    sims
end 

using JLD2
function main(;N=100,harvest = collect(0.0:1.0:50),T=500,NMC=250,burnin=50)
    
    # set parameters
    MSY = 10
    Fmsy = 0.2
    SigmaN = [0.05;;]
    sigma_a = [0.1,0.2,0.4]
    sigma_p = 2.0
    c1 = 5.0
    C2 = [0.0,0.02,0.04]
    C3 = [0.0, 2.0, 9.0]
    b = Fmsy
    maxV = [0.0,MSY]
    discount = 0.05
    
    # set value and policy functions 
    pars = (MSY,Fmsy,SigmaN,sigma_a,sigma_p,c1,C2[1],C3[1],b,maxV[1],discount)
    R,V,P,mod,delta=set_up_model(pars;N=N,harvest=harvest)
    
    grids = []
    pars_grid = []
    for sigma in sigma_a
        pars = (MSY,Fmsy,SigmaN,sigma,sigma_p,c1,C2[1],C3[1],b,maxV[1],discount)
        push!(pars_grid,pars)
        R,V,P,mod,delta=set_up_model(pars;N=N,harvest=harvest)
        push!(grids,compute_state_transitions(V,mod))     
    end 
    
    rewards = []
    pars_rewards = []
    for c in Iterators.product(C2,C3,maxV)
        c2,c3,ncv = c
        pars = (MSY,Fmsy,SigmaN,sigma_a[1],sigma_p,c1,c2,c3,b,ncv,discount)
        push!(pars_rewards,pars)
        R,V,P,mod,delta=set_up_model(pars;N=N,harvest=harvest)
        push!(rewards,compute_rewards(R,V,mod))  
    end 
    
    for i in 1:length(grids) 
        for j in 1:length(rewards)
            # Unpack parameters 
            pg = pars_grid[i]
            pr = pars_rewards[j]
            pars = (pg[1],pg[2],pg[3],pg[4],pg[5],pr[6],pr[7],pr[8],pr[9],pr[10],pr[11])
            
            # Run model 
            R,V,P,mod,delta=set_up_model(pars;N=N,harvest=harvest)
            V_ = VFI(grids[i],rewards[j], V, delta)
            P_ = compute_Policy(grids[i],rewards[j], V_, P,delta)
            
            # smulatin results
            dat = simulation(mod,P_,R,pars;T=T,NMC=NMC,burnin=burnin)
            
            # save results 
            sol = solution(pars,P_,V_,dat)
            @save string("/home/jhbuckne/SurplusProductionModel/FARM/data/Solution_",i, "_",j,".csv") sol
        end 
    end 
    
end 

if abspath(PROGRAM_FILE) == @__FILE__
    main(N=100,harvest=vcat(collect(0.0:0.25:15), collect(16:1.0:50)),T=1000,NMC=500)
    #main(N=2,harvest=[0,2.0],T=100,NMC=5)
end

end # module