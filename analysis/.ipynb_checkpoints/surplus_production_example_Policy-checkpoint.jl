include("SurplusProduction.jl")
include("BaseParams.jl")
using CSV
using Tables
using Sobol

function main()
    
    MSY = BaseParams.MSY
    Fmsy = BaseParams.Fmsy
    
    model=SurplusProduction.init_simpler( 
                BaseParams.Fmsy, # fishing mortaltiy rate at maximum sustainable yield
                BaseParams.buffer, # p-star
                BaseParams.tau, # process noice
                BaseParams.sigma_a, # active monitoring noise
                BaseParams.sigma_p, # passive monitoring noise
                BaseParams.H_weight, # harvets weight
                BaseParams.NCV_weight, # nonconsumptive values weight
                BaseParams.c1, # stock dependent costs 
                BaseParams.c2, # saturating / nonlinear costs
                BaseParams.b, # nonconsumptive values risk aversion 
                BaseParams.discount;
                mQuad = 25,N =100,CVmax = 1.5,
                Bmax = 3.0,Bmin = 0.025,threashold = 10^-5)

    
    Bhat = (1:(0.025*2*MSY/Fmsy):(2*MSY/Fmsy))
    CV = 0.01:0.025:1.01

    acc = zeros(length(Bhat)*length(CV),3)
    n=0
    for i in 1:length(Bhat)
        for j in 1:length(CV)
            n+=1
            Var = log(CV[j]^2+1)
            mu = log(Bhat[i]) - 0.5*Var 
            acc[n,:] = [Bhat[i],CV[j],model.Policy([mu,Var])]
        end
    end 
    
    CSV.write("/Users/johnbuckner/github/KalmanFilterPOMDPs/FARM/data/example_policy.csv",Tables.table(acc);sep=',')

end 


using CSV
using Tables
main()
