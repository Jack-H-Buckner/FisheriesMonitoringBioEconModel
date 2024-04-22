module surplus_production_VoI

include("SurplusProduction.jl")
include("BaseParams.jl")
using CSV
using Tables
using Sobol

function EVoI!(acc,
        ind0,
        Fmsy,
        pstar,
        tau,
        sigma_a,
        sigma_p,
        H_weight,
        NCV_weight,
        c1,
        c2,
        b,
        discount,
        Nsamples
    )

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
            b, # nonconsumptive values risk aversion 
            discount;
            MSY = 10,
            monitoring_costs = 1.0)
    
    MSY = 10
    Bmsy = MSY/Fmsy
    Bmax =  2*Bmsy; Bmin = 0.05*Bmsy
    CVmin = 0.01;CVmax = 1.0
    
    seq = Sobol.SobolSeq(2)
    for n in 1:Nsamples
        x = next!(seq)
        CV = CVmin + (CVmax - CVmin)*x[1]
        Bhat = Bmin + (Bmax - Bmin)*x[2]
        Var = log(CV^2+1)
        mu = log(Bhat) - 0.5*Var 
        acc[ind0+n,:] = [Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,
                            c1,c2,b,discount,Bhat/Bmsy,CV,model.VoI([mu,Var])]

    end 

    
    return acc
end 


function main()
    Nmc = 2^13 # 13
    Nsamples = 2^7 # 7
    dims = 14
    acc = zeros(Nsamples*Nmc,dims)
    s = Sobol.SobolSeq(dims)
    for i in 1:Nmc
        print(i," ")
        Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount=BaseParams.sample(s)
        EVoI!(acc,(i-1)*Nsamples,Fmsy,pstar,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount,Nsamples)
    end 
    return acc
end 

if abspath(PROGRAM_FILE) == @__FILE__
    print("here")
    dat = main()
    CSV.write("data/VoI.csv",Tables.table(dat);sep=',')
end


end # module