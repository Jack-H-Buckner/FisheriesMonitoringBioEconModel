include("SurplusProduction.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")
using Plots
using LaTeXStrings
using Random
using Distributions 
using Plots.PlotMeasures
# define model and solve MDP
function main()

    H_weights = [1.0,2.0,4.0,8.0,16.0,32.0,64.0] 
    CV_levels = [0.01,0.25,0.5,0.75]

    data = zeros(length(CV_levels),length(H_weights))
        
    data2 = zeros(length(CV_levels),length(H_weights))
        

    for i in 1:length(H_weights)
        model=SurplusProduction.init( 
                    BaseParams.Fmsy, # fishing mortaltiy rate at maximum sustainable yield
                    BaseParams.pstar, # p-star
                    BaseParams.tau, # process noice
                    BaseParams.sigma_a, # active monitoring noise
                    BaseParams.sigma_p, # passive monitoring noise
                    H_weights[i],#BaseParams.H_weight, # harvets weight
                    0.0,#BaseParams.NCV_weight, # nonconsumptive values weight
                    BaseParams.c1, # stock dependent costs 
                    BaseParams.c2, # saturating / nonlinear costs
                    BaseParams.b, # nonconsumptive values risk aversion 
                    BaseParams.discount;
                    MSY = BaseParams.MSY,
                    monitoring_costs = BaseParams.monitoring_costs,
                    N =75)

        j = 0
        for CV in CV_levels
            j+=1; B = BaseParams.MSY/BaseParams.Fmsy; sigma = log(CV^2+1)
            data[j,i] = 100 * model.VoI([log(B),sigma]) / BaseParams.H_weight
            data2[j,i] = 100 * model.VoI([log(B),sigma]) / model.Value([log(B),sigma])
        end 

    end 
    
    p1 = Plots.plot(size = (600,500), margin = 10mm)
    Plots.plot!( H_weights,data[1,:],label = L"CV = 0.01",c = Colors.RGB(0.8,0.6,0.2))
    Plots.plot!( H_weights,data[2,:],label = L"CV = 0.25",c = Colors.RGB(0.6,0.6,0.4))
    Plots.plot!( H_weights,data[3,:],label = L"CV = 0.50",c = Colors.RGB(0.4,0.6,0.6))
    Plots.plot!( H_weights,data[4,:],label = L"CV = 0.75",c = Colors.RGB(0.2,0.6,0.8),
                xlabel = L"\hat{B}_t/B_{MSY}", ylabel = string("Value of assessment \n (% of harvest value at ",L"B_{MSY}",")"),
                xtickfontsize=14,ytickfontsize=14,
                xguidefontsize=16,yguidefontsize=16,legendfontsize=14)
    savefig(p1,"~/github/KalmanFilterPOMDPs/examples/figures/percent_revenue_VoI.png")
    
    
    p2 = Plots.plot(size = (500,500), margin = 10mm)
    Plots.plot!( H_weights,data2[1,:],label = L"CV = 0.01",c = Colors.RGB(0.8,0.6,0.2))
    Plots.plot!( H_weights,data2[2,:],label = L"CV = 0.25",c = Colors.RGB(0.6,0.6,0.4))
    Plots.plot!( H_weights,data2[3,:],label = L"CV = 0.50",c = Colors.RGB(0.4,0.6,0.6))
    Plots.plot!( H_weights,data2[4,:],label = L"CV = 0.75",c = Colors.RGB(0.2,0.6,0.8),
                xlabel = L"\hat{B}_t/B_{MSY}", ylabel = string("Value of assessment \n (% of harvest value at ",L"B_{MSY}",")"),
                xtickfontsize=14,ytickfontsize=14,
                xguidefontsize=16,yguidefontsize=16,legendfontsize=14)
    savefig(p2,"~/github/KalmanFilterPOMDPs/examples/figures/percent_NPV_VoI.png")
        
end


main()