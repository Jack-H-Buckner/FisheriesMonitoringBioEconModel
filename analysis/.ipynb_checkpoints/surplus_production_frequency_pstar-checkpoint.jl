include("SurplusProduction.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")

using CSV
using Tables
using Distributions
using Sobol

function assessment_frequency!(Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount)

    model = SurplusProduction.init_pstar(
                Fmsy,pstar, tau, sigma_a, sigma_p, H_weight, NCV_weight, 
                c1, c2, b, discount;mQuad = 20,N =50,CVmax = 1.5,MSY = 10,
                monitoring_costs = 1.0,Bmax = 3.0,Bmin = 0.05,threashold = 10^-4.0)
    
    
    MSY = 10.0
    models = broadcast(i->deepcopy(model),1:Threads.nthreads())
    monitor = zeros(Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        x0 = [log(MSY/Fmsy)];s0 = ([log(MSY/Fmsy)], [0.3;;]);T = 750

        quad = BeliefStateTransitions.MvGaussHermite.init_mutable(10,s0[1],s0[2])

        dat=BeliefStateTransitions.simulation_kf(x0,s0,T,models[i].Policy,models[i].mod,models[i].Returns,quad)
    
        inds = collect(1:T)#[broadcast(i -> exp.(dat[2][i][1][1]), 1:T) .> 0.5*MSY/Fmsy]
        monitor[i] = sum(broadcast(i -> dat[3][i], inds) .- 1)/length(inds)*1/Threads.nthreads()
        
    end 
    
    return sum(monitor)
end



function main()
    Nmc = 2^9 #13
    dims = 11
    acc = zeros(Nmc,dims+1)
    s = Sobol.SobolSeq(dims)
    for n in 1:Nmc
        print(n," ")
        Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount=BaseParams.sample(s)
        omega = assessment_frequency!(Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount)
        acc[n,:] = [Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount,omega]
    end 
    return acc
end 


using CSV
using Tables

dat = main()
CSV.write("data/assessment_frequency_pstar.csv",Tables.table(dat);sep=',')


