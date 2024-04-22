
include("SurplusProduction.jl")
include("BaseParams.jl")
using CSV
using Tables
using Sobol

function EVoI!(acc,ind0,Fmsy,buffer,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount,Nsamples)

    model=SurplusProduction.init_simpler( 
            Fmsy, buffer, tau, sigma_a, sigma_p, H_weight, NCV_weight, c1, c2, b, discount;
            N = 50, CVmax = 1.5, Bmax = 3.5, Bmin = 0.05, threashold = 10^-5.0)
    
    model_no_montoring=SurplusProduction.init_simpler( 
            Fmsy, buffer, tau, sigma_a, sigma_p, H_weight, NCV_weight, c1, c2, b, discount;
            N = 50, CVmax = 1.5, Bmax = 3.5, Bmin = 0.05, threashold = 10^-5.0,actions = [1])

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
        value = model.Value([mu,Var]) - model_no_montoring.Value([mu,Var])
        acc[ind0+n,:] = [Fmsy,buffer,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount,Bhat/Bmsy,CV,value]

    end 

    return acc
end 


function main()
    Nmc = 2^9 # 13
    Nsamples = 2^6 # 7
    dims = 11
    acc = zeros(Nsamples*Nmc,dims+3)
    s = Sobol.SobolSeq(dims)
    for i in 1:Nmc
        print(i," ")
        Fmsy,buffer,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount=BaseParams.sample_simpler(s)
        EVoI!(acc,(i-1)*Nsamples,Fmsy,buffer,tau,sigma_a,sigma_p,H_weight,NCV_weight,c1,c2,b,discount,Nsamples)
    end 
    return acc
end 

dat = main()
CSV.write("data/total_VoI.csv",Tables.table(dat);sep=',')

