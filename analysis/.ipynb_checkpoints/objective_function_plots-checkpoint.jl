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
MSY = 10
Fmsy = 0.2

Harvest = 0.0:0.1:20

function plot_Harvest_values(Biomass,c1,c2;xlab = false, ylab =false, legendpos = false)
    observation = 1.0
    H_weight = 1.0
    NCV_weight = 0.0
    b = BaseParams.b
    monitoring_cost = 0.0
    MSY = BaseParams.MSY 
    Fmsy = BaseParams.Fmsy
    Harvest = 0.0:0.01:2
    MSY = BaseParams.MSY
    styles = [:solid,:dash,:dashdot,:dot,:solid,:dash,:dashdot,:dot,:solid,:dash,:dashdot,:dot]
    values = broadcast(H -> SurplusProduction.objective_function([log(Biomass[1])],
                        observation,H*MSY,H_weight, NCV_weight, 
                        c1,c2, b, monitoring_cost ,MSY,Fmsy ), Harvest)
    p1 = Plots.plot(Harvest,values, color = "black", linestyle = styles[1], 
                    width = 1.5,label = string(L"B = ", Biomass[1]),
                    legendtitle = "Biomass", 
                    legendposition = legendpos,
                    title = string(L"c_1 = ", c1, "  ", L"c_2 = ", c2),
                    ylims = [-1.0,2])
    Plots.hline!([0.0],linestyle = :dash, color = "black", label = false)
    
    for i in 2:length(Biomass)
        values = broadcast(H -> SurplusProduction.objective_function([log(Biomass[i])],
                        observation,H*MSY,H_weight, NCV_weight, 
                        c1,c2, b, monitoring_cost ,MSY,Fmsy ), Harvest)
        Plots.plot!(p1,Harvest,values, color = "black", linestyle = styles[i], 
                        width = 1.5,label = string(L"B = ", Biomass[i]))
    end 
    
    if ylab 
        Plots.plot!(p1,ylabel = string("fishery value ", L"(\pi_{H})"))   
    end
    
        
    if xlab
        Plots.plot!(p1,xlabel = string("Harvest ", L"(H_t/MSY)") )
    end
    
    return p1
end

Biomass = [25,50,75]
p1 = plot_Harvest_values(Biomass,BaseParams.lc1,BaseParams.lc2, ylab = true, legendpos = :bottomright)
p2 = plot_Harvest_values(Biomass,BaseParams.lc1,BaseParams.uc2)
p3 = plot_Harvest_values(Biomass,BaseParams.uc1,BaseParams.lc2, ylab = true, xlab = true)
p4 = plot_Harvest_values(Biomass,BaseParams.uc1,BaseParams.uc2, xlab = true)

plot(p1,p2,p3,p4)

savefig(plot(p1,p2,p3,p4),"../examples/figures/harvest_values.png")



### Nonconsumptive values

x = [log(50)]
observation = 1
harvest = 0.0
H_weight = 0.0
NCV_weight = 1.0
c1 = 0.0
c2 = 0.0
monitoring_cost = 0.0
MSY = BaseParams.MSY 
Fmsy = BaseParams.Fmsy 


Biomass = 0.0:0.01:2
values_b11 = broadcast(B -> SurplusProduction.objective_function([log(B*MSY/Fmsy)],observation,0.0,H_weight, NCV_weight, c1,c2, BaseParams.lb, monitoring_cost,MSY, Fmsy ), Biomass)
values_b12 = broadcast(B -> SurplusProduction.objective_function([log(B*MSY/Fmsy)],observation,0.0,H_weight, NCV_weight, c1,c2, BaseParams.b, monitoring_cost,MSY, Fmsy ), Biomass)
values_b13 = broadcast(B -> SurplusProduction.objective_function([log(B*MSY/Fmsy)],observation,0.0,H_weight, NCV_weight, c1,c2, BaseParams.ub, monitoring_cost,MSY, Fmsy ), Biomass)
p5 = Plots.plot(Biomass,values_b11, color = "black", linestyle = :dashdot, width = 2,label = L"B = 0.3")
Plots.plot!(Biomass,values_b12, color = "black", linestyle = :dash, width = 2,label = L"B = 0.5")
Plots.plot!(Biomass,values_b13, color = "black",width = 2,label = L"b = 0.8",
            ylabel = string("nonconsumptive \n value " ,L"(\pi_{nc})"), xlabel = L"B/B_{MSY}")

plot!(size = (350,250))
savefig(p5,"../examples/figures/nonconsmptive_values.png")

# harvest and biomass plot 

MSY = BaseParams.MSY
Fmsy = BaseParams.Fmsy
observation = 1.0
NCV_weight = BaseParams.NCV_weight
H_weight = BaseParams.H_weight
c1 = BaseParams.c1
c2 = BaseParams.c2
b = BaseParams.b
monitoring_cost = 0.0

f(x, y) = SurplusProduction.objective_function([log(x*MSY/Fmsy)],observation,y*MSY,H_weight, NCV_weight, c1,c2, b, monitoring_cost,MSY, Fmsy )

x = range(0.05, 2, length=500)
y = range(0.05, 2, length=500)
z = @. f(x', y)
z[(y*MSY)./ (x*MSY/Fmsy)' .> 0.9] .= z[argmin(z)]

p6 =heatmap(x, y, z,
xlabel = string("Biomass ", L"(\hat{B}_t/B_{MSY})"), 
ylabel = string("Harvest ", L"(\hat{H_t}/MSY)"),
title = "Objective function")
contour!(x, y, z,color = "white", levels = [25,20,15,10,5,0.0])

plot!(size = (350,250))
savefig(p6,"../examples/figures/base_objective_function.png")


