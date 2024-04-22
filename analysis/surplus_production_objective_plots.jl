using Plots
using LaTeXStrings
include("SurplusProduction.jl")
include("BaseParams.jl")
x = [log(50)]
observation = 1
harvest = 10
H_weight = 1.0
NCV_weight = 0.0
c1 = 0.0
c2 = 0.0
NCVshape =0.2
monitoring_cost = 0.0


Harvest = 0.0:0.1:20
values_B25 = broadcast(H -> SurplusProduction.objective_function([log(25)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B50 = broadcast(H -> SurplusProduction.objective_function([log(50)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B75 = broadcast(H -> SurplusProduction.objective_function([log(75)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY,BaseParams.Fmsy ), Harvest)
p1 = Plots.plot(Harvest,values_B25, color = "black", linestyle = :dashdot, width = 2,label = L"B = 25")
Plots.plot!(Harvest,values_B50, color = "black", linestyle = :dash, width = 2,label = L"B = 50")
Plots.plot!(Harvest,values_B75, ylim = [0,2], color = "black", width = 2,label = L"B = 75",
            ylabel = string("fishery profit ", L"(\pi_{H})"), legendtitle = "Biomass", 
            title = string(L"c_1 = 0", "  ", L"c_2 = 0"))

c2 = 0.2
c1 = 0.0
values_B25 = broadcast(H -> SurplusProduction.objective_function([log(25)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B50 = broadcast(H -> SurplusProduction.objective_function([log(50)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B75 = broadcast(H -> SurplusProduction.objective_function([log(75)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
p2 = Plots.plot(Harvest,values_B25, color = "black", linestyle = :dashdot, width = 2)
Plots.plot!(Harvest,values_B50, color = "black", linestyle = :dash, width = 2)
Plots.plot!(Harvest,values_B75, ylim = [0,2], color = "black", width = 2, legend = false, 
            title = string(L"c_1 = 0", "  ", L"c_2 = 0.2"))

c1 = 0.5
values_B25 = broadcast(H -> SurplusProduction.objective_function([log(25)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B50 = broadcast(H -> SurplusProduction.objective_function([log(50)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B75 = broadcast(H -> SurplusProduction.objective_function([log(75)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
p3 = Plots.plot(Harvest,values_B25, color = "black", linestyle = :dashdot, width = 2)
Plots.plot!(Harvest,values_B50, color = "black", linestyle = :dash, width = 2)
Plots.plot!(Harvest,values_B75, ylim = [0,2], color = "black", width = 2, legend = false,
            ylabel = string("fishery profit ", L"(\pi_{H})"), xlabel = "Harvest",
            title = string(L"c_1 = 0.5", "  ", L"c_2 = 0.0"))


c2 = 0.1
c1 = 0.25
values_B25 = broadcast(H -> SurplusProduction.objective_function([log(25)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B50 = broadcast(H -> SurplusProduction.objective_function([log(50)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
values_B75 = broadcast(H -> SurplusProduction.objective_function([log(75)],observation,H,H_weight, NCV_weight, c1,c2, 
                        NCVshape, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Harvest)
p4 = Plots.plot(Harvest,values_B25, color = "black", linestyle = :dashdot, width = 2)
Plots.plot!(Harvest,values_B50, color = "black", linestyle = :dash, width = 2)
Plots.plot!(Harvest,values_B75, ylim = [0,2], color = "black", width = 2, legend = false,
            xlabel = "Harvest", title = string(L"c_1 = 0.25", "  ", L"c_2 = 0.1"))
savefig(plot(p1,p2,p3, p4),string("../examples/figures/harvest_objective.png"))




x = [log(50)]
observation = 1
harvest = 0.0
H_weight = 0.0
NCV_weight = 1.0
c1 = 0.0
c2 = 0.0
monitoring_cost = 0.0


Biomass = 0.0:0.01:2
values_b11 = broadcast(B -> SurplusProduction.objective_function([log(B*MSY/Fmsy)],observation,0.0,H_weight, NCV_weight, c1,c2, 
                        0.2, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Biomass)
values_b12 = broadcast(B -> SurplusProduction.objective_function([log(B*MSY/Fmsy)],observation,0.0,H_weight, NCV_weight, c1,c2, 
                        0.5, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Biomass)
values_b13 = broadcast(B -> SurplusProduction.objective_function([log(B*MSY/Fmsy)],observation,0.0,H_weight, NCV_weight, c1,c2, 
                        0.8, monitoring_cost,BaseParams.MSY, BaseParams.Fmsy ), Biomass)
p5 = Plots.plot(Biomass,values_b11, color = "black", linestyle = :dashdot, width = 2,label = L"b = 0.2")
Plots.plot!(Biomass,values_b12, color = "black", linestyle = :dash, width = 2,label = L"b = 0.5")
Plots.plot!(Biomass,values_b13, color = "black",width = 2,label = L"b = 0.8",
            ylabel = string("nonconsumptive value " ,L"(\pi_{nc})"), xlabel = L"B/B_{MSY}",
            xguidefontsize=18,yguidefontsize=17,
            xtickfontsize=14,ytickfontsize=14,legendfontsize=18)

savefig(p5,string("../examples/figures/nonconsumptive_objective.png"))
