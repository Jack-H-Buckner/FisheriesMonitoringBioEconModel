# this file runs closed loop simulations to test the performance of alternative polcies
# import Pkg; Pkg.add("NLsolve");Pkg.add("FastGaussQuadrature"); Pkg.add("Interpolations")
# Pkg.add("KalmanFilters"); Pkg.add("TickTock"); Pkg.add("StatsBase");Pkg.add("CSV")
# Pkg.add("Tables")
include("SurplusProduction.jl")
include("../src/ParticleFilters.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")
using Distributions

println(Threads.nthreads())
    


model=SurplusProduction.init( 
            BaseParams.Fmsy, # fishing mortaltiy rate at maximum sustainable yield
            BaseParams.pstar, # p-star
            BaseParams.tau, # process noice
            BaseParams.sigma_a, # active monitoring noise
            BaseParams.sigma_p, # passive monitoring noise
            BaseParams.H_weight, # harvets weight
            0.0, #BaseParams.NCV_weight, # nonconsumptive values weight
            BaseParams.c1, # stock dependent costs 
            BaseParams.c2, # saturating / nonlinear costs
            BaseParams.b, # nonconsumptive values risk aversion 
            BaseParams.discount;
            MSY = BaseParams.MSY,
            monitoring_costs = BaseParams.monitoring_costs,
            N =100)


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

# find CV value                       
function argswitch(row,observations,values)
    row = observations[row,:]
    for i in 1:length(row)
        if row[i] == 2.0
            return values[i]
        end 
    end 
    return values[end]
end 
function minswitch(observations,values)
    acc = zeros(100)
    for i in 1:100
        acc[i] = argswitch(i,observations,values)
    end
    return acc[argmin(acc)]
end 
                                            
                                            
Var_min = minswitch(model.Policy.observations,model.Policy.nodesVar)
CVmin = sqrt(exp(Var_min)-1)
policy1=  s -> model.Policy([s[1][1],s[2][1,1]])
policy2 = s -> steps(s, log.([25.0,60.0]), [Var_min,Var_min,Var_min])
policy3 = s -> steps(s, log.([25.0,65.0]), [log((0.5*CVmin)^2+1),Var_min,Var_min])
policy4 = s -> steps(s, log.([25.0,65.0]), [log((0.5*CVmin)^2+1),Var_min,log((1.5*CVmin)^2+1)])
Policies = [policy1,policy2,policy3,policy4]  

# Policy plot data 
function Policy_plot_data(Policies)
    n=0
    CV_vals = collect(0.01:0.005:1.0)
    Bhat_vals = 1:1.0:100
    acc = zeros(4*length(Bhat_vals)*length(CV_vals),4)
    for i in 1:length(Policies)
        for j in 1:length(Bhat_vals)
            for k in 1:length(CV_vals)
                Var = log(CV_vals[k]^2+1)
                mu = log(Bhat_vals[j]) - 0.5*Var 
                n += 1
                acc[n,1] = Bhat_vals[j]
                acc[n,2] = CV_vals[k]
                acc[n,3] = i
                acc[n,4] = Policies[i](([mu],[Var;;]))
            end
        end
    end
    return acc 
end 

using DelimitedFiles
writedlm( "data/alternative_policies_map.csv",  Policy_plot_data(Policies), ',') 


B0 = [15,50,110]
sigma0 = [log(0.25^2+1),log(0.5^2+1),log(0.75^2+1)]
                 
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
results = main(model,Policies,BaseParams.discount,B0,sigma0)
CSV.write("data/alt_policies_performance.csv_NCV_0",Tables.table(results);sep=',')
