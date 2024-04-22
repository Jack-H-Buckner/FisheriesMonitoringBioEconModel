include("SurplusProduction.jl")
include("../src/BeliefStateTransitions.jl")
include("BaseParams.jl")
using Plots
using LaTeXStrings
using Random
using Distributions 
# define model and solve MDP
function main()
    Random.seed!(1234)
    MSY = BaseParams.MSY
    Fmsy = BaseParams.Fmsy
    model=SurplusProduction.init( 
                BaseParams.Fmsy, # fishing mortaltiy rate at maximum sustainable yield
                BaseParams.pstar, # p-star
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
    
    policy =  s -> model.Policy([s[1][1],s[2][1,1]])

    # initial conditions for simulation  
    B0 = MSY/Fmsy
    sigma0 = 0.25
    
    # lenth of simualtion 
    T = 50
    
    # numer fo particle sin beleif state aproximation 
    number_of_particles = 1000

    filter = BeliefStateTransitions.ParticleFilters.init(number_of_particles,Distributions.MvNormal([log(B0)],[sigma0;;]))
    x0 = [rand(Distributions.Normal(log(B0), sqrt(sigma0)))]
    s0 = ([log(B0)],[sigma0;;])

    dat = BeliefStateTransitions.simulation(x0,s0,T,filter,policy,model.mod,model.Returns)
    
    p1 = Plots.scatter(broadcast(i -> exp(dat[1][i][1]), 1:T), markersize= 3, label = string("true biomass: ",L"B_t"),c=3)
    Plots.plot!(broadcast(i -> exp.(dat[2][i][1][1]), 1:T), color = "black", label = string("estimated biomass: ",L"\hat{B}_t"))
    Plots.plot!(broadcast(i -> exp.(dat[2][i][1][1] + 2*sqrt(dat[2][i][2][1])), 1:T), 
                fillrange = broadcast(i -> exp.(dat[2][i][1][1] - 2*sqrt(dat[2][i][2][1])), 1:T), 
                alpha =0.2, color = "grey", label = string("credible interval ", L"(2\sigma)") , ylab = "Biomass")
    Plots.plot!(broadcast(i -> exp.(dat[2][i][1][1]- 2*sqrt(dat[2][i][2][1])), 1:T), color = "grey",
                alpha = 0.2, label = "")
    inds = collect(1:T)[broadcast(i -> dat[3][i], 1:T) .== 2]
    y = zeros(length(inds))
    inds_0 = collect(1:T)[broadcast(i -> dat[3][i], 1:T) .== 1]
    y_0 = zeros(length(inds))
    Plots.plot!(broadcast(i -> dat[4][i], 1:T), color = "black", label = string("harvest: ", L"H_t"), 
    xlabel = "Time",linestyle = :dash)
    Plots.scatter!(inds,y,markershape = :utriangle, color = "black",markersize = 4,
                    label = string("monitoring: ", L"\sigma = \sigma_{1}"))
    Plots.scatter!(inds_0,y_0,markershape = :utriangle, color = "white",markersize = 4,
                    label = string("not monitoring: ", L"\sigma = \sigma_{0}"),
                    xtickfontsize=12, ytickfontsize=12, guidefont=18, legendfont=12,
                    ylims = (-5.0,225.0))

    savefig(p1,"~/github/KalmanFilterPOMDPs/examples/figures/example_simulation.png")
end
    
main()