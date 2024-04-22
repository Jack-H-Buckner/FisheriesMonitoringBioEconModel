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
                BaseParams.NCV_weight, # nonconsumptive values weight
                BaseParams.c1, # stock dependent costs 
                BaseParams.c2, # saturating / nonlinear costs
                BaseParams.b, # nonconsumptive values risk aversion 
                BaseParams.discount;
                MSY = BaseParams.MSY,
                monitoring_costs = BaseParams.monitoring_costs,
                N =50, # 50
                threashold= 10^-5)


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
              
    return sum(acc)/length(acc)
        
end 
                                            


function main(model,policy,discount,B0,sigma0;NMC = 10000,T=100)
                            
    delta = 1/(1+discount)
    simulated_value = zeros(9)
    value_function = zeros(9)
    num_iter=0                     
    for init in Iterators.product(B0,sigma0)
        num_iter+=1
        #simulated_value[num_iter] = ENPV(init[1],init[2],policy,delta,T,NMC,model.mod,model.Returns;Nfilter = 2000)
        value_function[num_iter] = model.Value([log(init[1]),init[2]])
    end 
    return simulated_value, value_function
end 

using CSV
using Tables

                                
policy=  s -> model.Policy([s[1][1],s[2][1,1]])
B0 = [15,50,110]
sigma0 = [log(0.25^2+1),log(0.5^2+1),log(0.75^2+1)]
                 
simulated_value, value_function = main(model,policy,BaseParams.discount,B0,sigma0)

# using Plots                             
# p1 = Plots.scatter(simulated_value, value_function)
# mn = [simulated_value[argmin(simulated_value)],value_function[argmin(value_function)]]
# mn = mn[argmin(mn)]
# mx = [simulated_value[argmax(simulated_value)],value_function[argmax(value_function)]]
# mx = mx[argmax(mx)]
# Plots.plot!([mn,mx],[mn,mx], xlabel = "simulated", ylabel = "dynamic programming")
# savefig(p1,"data/VoI_ENPV_test.png")   
                                
#CSV.write("FARM/data/simulated_values.csv",Tables.table(simulated_value);sep=',')
CSV.write("data/value_function.csv",Tables.table(value_function);sep=',')
                                
