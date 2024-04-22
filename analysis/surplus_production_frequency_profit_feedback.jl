include("SurplusProduction.jl")
include("surplus_production_VoI.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")

using CSV
using Tables
using Distributions
using Sobol

function assessment_frequency!(Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount)

    model = SurplusProduction.init_profit_feedback(
            Fmsy, buffer, tau, sigma_a, sigma_p, NCV_weight, price, BaseParams.c, b,discount;
            mQuad = 20,N =50,CVmax = 1.0,Bmax = 3.0,Bmin = 0.05, threashold = 10^-5.0)
    
    
    MSY = 10.0
    models = broadcast(i->deepcopy(model),1:Threads.nthreads())
    monitor = zeros(Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        x0 = [log(MSY/Fmsy)];s0 = ([log(MSY/Fmsy)], [0.3;;]);T = 500

        quad = BeliefStateTransitions.MvGaussHermite.init_mutable(10,s0[1],s0[2])

        dat=BeliefStateTransitions.simulation_kf(x0,s0,T,models[i].Policy,models[i].mod,models[i].Returns,quad)
    
        inds = collect(1:T)#[broadcast(i -> exp.(dat[2][i][1][1]), 1:T) .> 0.5*MSY/Fmsy]
        monitor[i] = sum(broadcast(i -> dat[3][i], inds) .- 1)/length(inds)*1/Threads.nthreads()
        
    end 
    
    return sum(monitor)
end



function main()
    Nmc = 2^11 #13
    dims = 9
    acc = zeros(Nmc,dims+1)
    s = Sobol.SobolSeq(dims)
    for n in 1:Nmc
        print(n, " ")
        Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount=BaseParams.sample_profit_feedback(s)
        omega = assessment_frequency!(Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount)
        acc[n,:] = [Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount,omega]
    end 
    return acc
end 


using CSV
using Tables

dat = main()
CSV.write("data/assessment_frequency_profit_feedback.csv",Tables.table(dat);sep=',')


