include("SurplusProduction.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")
using Plots
using LaTeXStrings
using Random
using Distributions 
using Plots.PlotMeasures
# define model and solve MDP
function main(pstar)
    MSY = BaseParams.MSY
    Fmsy = BaseParams.Fmsy
    model=SurplusProduction.init( 
                BaseParams.Fmsy, # fishing mortaltiy rate at maximum sustainable yield
                pstar, # p-star
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
                N =50)
    
    CV_levels = [0.1,0.25,0.5,0.75]
    BBmsy_levels = collect(0.01:0.01:2.0)
    data = zeros(length(CV_levels),length(BBmsy_levels))
    data_effort = zeros(length(CV_levels),length(BBmsy_levels))
    i = 0
    for CV in CV_levels
        j = 0; i+=1
        for BBmsy in BBmsy_levels
            j+=1; B = MSY*BBmsy/Fmsy; sigma = log(CV^2+1)
            state = ([log(B)],[sigma;;])
            #println(state)
            data[i,j] = model.mod.fixed_control(state) / MSY
            data_effort[i,j] = model.mod.fixed_control(state) / B 
        end
    end 

    p1 = Plots.plot(size = (600,500), margin = 10mm)
    Plots.plot!(BBmsy_levels,data[1,:],label = L"CV = 0.1",c = Colors.RGB(0.8,0.6,0.2))
    Plots.plot!(BBmsy_levels,data[2,:],label = L"CV = 0.25",c = Colors.RGB(0.6,0.6,0.4))
    Plots.plot!(BBmsy_levels,data[3,:],label = L"CV = 0.50",c = Colors.RGB(0.4,0.6,0.6))
    Plots.plot!(BBmsy_levels,data[4,:],label = L"CV = 0.75",c = Colors.RGB(0.2,0.6,0.8),
                xlabel = L"\hat{B}_t/B_{MSY}", ylabel = string("Harvest ",L"(H_t / MSY)"),
                xtickfontsize=14,ytickfontsize=14,
                xguidefontsize=16,yguidefontsize=16,legendfontsize=14)
    savefig(p1,string("~/github/KalmanFilterPOMDPs/examples/figures/example_HCR_pstar_", pstar,".png"))

    p1 = Plots.plot(size = (600,500), margin = 10mm)
    Plots.plot!(BBmsy_levels,data_effort[1,:],label = L"CV = 0.1",c = Colors.RGB(0.8,0.6,0.2))
    Plots.plot!(BBmsy_levels,data_effort[2,:],label = L"CV = 0.25",c = Colors.RGB(0.6,0.6,0.4))
    Plots.plot!(BBmsy_levels,data_effort[3,:],label = L"CV = 0.50",c = Colors.RGB(0.4,0.6,0.6))
    Plots.plot!(BBmsy_levels,data_effort[4,:],label = L"CV = 0.75",c = Colors.RGB(0.2,0.6,0.8),
                xlabel = L"\hat{B}_t/B_{MSY}", ylabel = string("Target effort ", L"(H_t / \hat{B}_t)"),
                xtickfontsize=14,ytickfontsize=14,
                xguidefontsize=16,yguidefontsize=16,legendfontsize=14)
    savefig(p1,string("~/github/KalmanFilterPOMDPs/examples/figures/example_HCR_effort_pstar_", pstar,".png"))

end
    
main(0.4)
main(0.3)
main(0.5)