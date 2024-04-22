include("SurplusProduction.jl")
include("surplus_production_VoI.jl")
include("BaseParams.jl")
using CSV
using Tables
using Sobol

function Policy!(acc,ind0,Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,NCVshape,discount)

    model=SurplusProduction.init( 
            Fmsy, # fishing mortaltiy rate at maximum sustainable yield
            pstar, # p-star
            tau, # process noice
            sigma_a, # active monitoring noise
            sigma_p, # passive monitoring noise
            H_weight, # harvets weight
            NCV_weight, # nonconsumptive values weight
            c1, # stock dependent costs 
            c2, # saturating / nonlinear costs
            NCVshape, # nonconsumptive values risk aversion 
            discount;
            MSY = 10,
            monitoring_costs = 1.0
        )
    
    Bhat = (1:(0.1*2*MSY/Fmsy):(2*MSY/Fmsy))
    CV = 0.01:0.1:1.0

    
    n=0
    for i in 1:length(Bhat)
        for j in 1:length(CV)
            n+=1
            Var = log(CV[j]^2+1)
            mu = log(Bhat[i]) - 0.5*Var 
            acc[ind0+n,:] = [sigma_a,sigma_p,SigmaN,Fmsy,NMVmax,price,Bhat[i],CV[j],model.Policy([mu,Var])]
        end
    end 
    
    for i in 1:400
        Bhat= 2*MSY*rand()/Fmsy
        CV =rand()*1.0
        Var = log(CV^2+1)
        mu = log(Bhat) - 0.5*Var 
        acc[ind0+n+i,:] = [sigma_a,sigma_p,SigmaN,Fmsy,NMVmax,price,Bhat,CV,model.Policy([mu,Var])]
    end 
    
    return acc
end 

function main()
    Nmc = 1000
    acc = zeros(500*Nmc,9)
    s = Sobol.SobolSeq(dims)
    for i in 1:Nmc
        print(i," ")
        Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,NCVshape,discount=BaseParams.sample(s)
        Policy!(acc,(i-1)*500,Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,NCVshape,discount)
    end 
    return acc
end 


using CSV
using Tables
dat = main()
CSV.write("data/Policies.csv",Tables.table(dat);sep=',')