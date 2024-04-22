include("SurplusProduction.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")
using Plots
using LaTeXStrings
using Random
using Distributions 
using Plots.PlotMeasures
# define model and solve MDP
function main(H_weight)

    model=SurplusProduction.init( 
                BaseParams.Fmsy, # fishing mortaltiy rate at maximum sustainable yield
                BaseParams.pstar, # p-star
                BaseParams.tau, # process noice
                BaseParams.sigma_a, # active monitoring noise
                BaseParams.sigma_p, # passive monitoring noise
                H_weight,#BaseParams.H_weight, # harvets weight
                0.0,#BaseParams.NCV_weight, # nonconsumptive values weight
                BaseParams.c1, # stock dependent costs 
                BaseParams.c2, # saturating / nonlinear costs
                BaseParams.b, # nonconsumptive values risk aversion 
                BaseParams.discount;
                MSY = BaseParams.MSY,
                monitoring_costs = BaseParams.monitoring_costs,
                N =50)
    MSY = BaseParams.MSY
    Fmsy = BaseParams.Fmsy
    CV_levels = [0.01,0.25,0.5,0.75]
    BBmsy_levels = collect(0:0.01:2.0)
    data = zeros(length(CV_levels),length(BBmsy_levels))
    data2 = zeros(length(CV_levels),length(BBmsy_levels)) 
    i = 0
    for CV in CV_levels
        j = 0; i+=1
        for BBmsy in BBmsy_levels
            j+=1; B = BaseParams.MSY*BBmsy/BaseParams.Fmsy; sigma = log(CV^2+1)
            data[i,j] = 100 * model.VoI([log(B),sigma]) / BaseParams.H_weight
            data2[i,j] = 100 * model.VoI([log(B),sigma]) / model.Value([log(B),sigma])
        end
    end 

    p1 = Plots.plot(size = (500,500), margin = 10mm)
    Plots.plot!(BBmsy_levels,data[1,:],label = L"CV = 0.01",c = Colors.RGB(0.8,0.6,0.2))
    Plots.plot!(BBmsy_levels,data[2,:],label = L"CV = 0.25",c = Colors.RGB(0.6,0.6,0.4))
    Plots.plot!(BBmsy_levels,data[3,:],label = L"CV = 0.50",c = Colors.RGB(0.4,0.6,0.6))
    Plots.plot!(BBmsy_levels,data[4,:],label = L"CV = 0.75",c = Colors.RGB(0.2,0.6,0.8),
                xlabel = L"\hat{B}_t/B_{MSY}", ylabel = string("Value of assessment \n (% of harvest value at ",L"B_{MSY}",")"),
                xtickfontsize=14,ytickfontsize=14,
                xguidefontsize=16,yguidefontsize=16,legendfontsize=14)
    savefig(p1,string("~/github/KalmanFilterPOMDPs/examples/figures/example_VoI_", H_weight, ".png"))
    
    
    p2 = Plots.plot(size = (500,500), margin = 10mm)
    Plots.plot!(BBmsy_levels,data2[1,:],label = L"CV = 0.01",c = Colors.RGB(0.8,0.6,0.2))
    Plots.plot!(BBmsy_levels,data2[2,:],label = L"CV = 0.25",c = Colors.RGB(0.6,0.6,0.4))
    Plots.plot!(BBmsy_levels,data2[3,:],label = L"CV = 0.50",c = Colors.RGB(0.4,0.6,0.6))
    Plots.plot!(BBmsy_levels,data2[4,:],label = L"CV = 0.75",c = Colors.RGB(0.2,0.6,0.8),
                xlabel = L"\hat{B}_t/B_{MSY}", ylabel = string("Value of assessment \n (% of harvest value at ",L"B_{MSY}",")"),
                xtickfontsize=14,ytickfontsize=14,
                xguidefontsize=16,yguidefontsize=16,legendfontsize=14, ylims = (0.0,3.0))
    savefig(p2,string("~/github/KalmanFilterPOMDPs/examples/figures/example_VoI_percent_NPV_", H_weight, ".png"))
end
    
main(10.0)    
main(15.0)
main(20.0)
main(25.0)
main(50.0)