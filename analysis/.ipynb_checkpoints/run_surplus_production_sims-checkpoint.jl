# this file runs closed loop simulations to test the performance of alternative polcies
include("SurplusProduction.jl")
include("../src/ParticleFilters.jl")
include("../src/BeliefStateTransitions.jl")
using Distributions

println(Threads.nthreads())
    
# define model and solve MDP
Fmsy= 0.2; price=1.0;sigma_a=0.1;sigma_p=1.5;NMVmax = 1.0;Sigma_N=0.05
MSY=10;pstar=0.4;c1 = 1.0/Fmsy;c2=0.0085
c3=2.0;b=2*Fmsy/MSY;discount=0.05;SigmaN=0.05
model=SurplusProduction.init_model(MSY,Fmsy,pstar,SigmaN,sigma_a,sigma_p,c1,c2,c3,b,
                                        NMVmax*MSY,discount;price=price,N=2,actions=[1,2],CVmax=1.05)


# define alternative policies 
function steps(s, Bhat_breaks, sigma_levels)
    
    if s[1][1] < Bhat_breaks[1]
        if s[2][1,1] > sigma_levels[1]
            return 2
        else
            return 1
        end 
    elseif s[1][1] <Bhat_breaks[2]
        if s[2][1,1] > sigma_levels[2]
            return 2
        else
            return 1
        end 
    else
        if s[2][1,1] > sigma_levels[3]
            return 2
        else
            return 1
        end 
    end 
    
end 


function NPV(B0, sigma0, P,delta, T, model, R;Nfilter = 500)
    
    filter = BeliefStateTransitions.ParticleFilters.init(Nfilter,Distributions.MvNormal([log(B0)],[sigma0;;]))#sigma0
    x0 = [rand(Distributions.Normal(log(B0), sqrt(sigma0)))]
    s0 = ([log(B0)],[sigma0;;])
    dat = BeliefStateTransitions.simulation(x0,s0,T,filter,P,model,R)
    if dat =="failed"
        return dat
    end 
        
    return sum(broadcast(t -> dat[5][t] * delta^t, 1:T))
    
end 
    
function ENPV(B0, sigma0, P,delta, T, NMC, model,  R;Nfilter = 500)
    
    acc = zeros(NMC)
            count = 0
    Threads.@threads for i in 1:NMC
        if mod(i,50) == 0
            print(i, " ")
        end 
        npv = NPV(B0, sigma0, P,delta, T, model, R;Nfilter = Nfilter)
        if npv != "failed"
            count += 1
            acc[i] =npv 
        else
            print("failed")   
        end
    end 
              
    return acc
        
end 
                        
policy1=  s -> model.Policy([s[1][1],s[2][1,1]])
policy2 = s -> steps(s, log.([25.0,50.0]), [log(0.388^2+1),log(0.388^2+1),log(0.388^2+1)])
policy3 = s -> steps(s, log.([25.0,55.0]), [log(0.2^2+1),log(0.388^2+1),log(0.388^2+1)])
policy4 = s -> steps(s, log.([25.0,55.0]), [log(0.2^2+1),log(0.388^2+1),log(0.55^2+1)])
Policies = [policy1,policy2,policy3,policy4]  
                        
B0 = [11,50,110]
sigma0 = [0.0606, 0.1315, 0.223]
                   
function main(model,policies,discount,B0,sigma0;NMC = 10000,T=100)
    delta = 1/(1+discount)
    
    acc = zeros(9*4,NMC)
    num_iter=0                     
    for init in Iterators.product(B0,sigma0)
        for p in Policies
            num_iter+=1
            acc[num_iter,:] = ENPV(init[1],init[2],p,delta,T,NMC,model.mod,model.Returns;Nfilter = 2000)
        end 
    end 
    return acc
end 

using CSV
using Tables
results = main(model,Policies,0.05,B0,sigma0)
CSV.write("data/alt_policies_performance.csv",Tables.table(results);sep=',')
