include("SurplusProduction.jl")
include("BaseParams.jl")

using CSV
using Tables
using Sobol

function EVoI!(acc,ind0,Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount,Nsamples)
    
    model = SurplusProduction.init_profit_feedback(
            Fmsy, buffer, tau, sigma_a, sigma_p, NCV_weight, price, BaseParams.c, b,discount;
            mQuad = 20,N =50,CVmax = 1.0,Bmax = 3.0,Bmin = 0.05, threashold = 10^-5.0)

    
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
        acc[ind0+n,:] = [Fmsy,buffer,tau,sigma_a,sigma_p,NCV_weight,price,b,discount,Bhat/Bmsy,CV,model.VoI([mu,Var])]

    end 
  
end 


function main()
    
    Nmc = 2^11 # 13
    Nsamples = 2^7 # 7
    dims = 9
    acc = zeros(Nsamples*Nmc,dims+3)
    s = Sobol.SobolSeq(dims)
    
    for i in 1:Nmc
        print(i," ")
        Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount=BaseParams.sample_profit_feedback(s)
        EVoI!(acc,(i-1)*Nsamples,Fmsy,buffer,tau,sigma_a,sigma_p,price,NCV_weight,b,discount,Nsamples)
    end 
    
    return acc
end 


dat = main()
CSV.write("data/VoI_profit_feedback.csv",Tables.table(dat);sep=',')

