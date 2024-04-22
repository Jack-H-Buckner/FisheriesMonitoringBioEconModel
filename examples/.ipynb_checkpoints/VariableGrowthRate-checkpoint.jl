module VariableGRowthRate

include("SurplusProduction.jl")

mutable struct parameters
    FMSY::Float64
    MSY::Float64
    auto_correlation::Float64
    transition_parameters::AbstractVector{Float64}
end 

function parameters(;FMSY = 0.2,MSY = 10,auto_correlation=0.9,
                    stock_process_errors_variance = 0.025, environmental_variance = 0.025,
                    observational_errors_variance = 0.1,
                    price = 1.0,cost_of_effort = 0.0,nonlinear_costs = 0.01,
                    over_fished_threshold = 0.25, over_fished_penelty = 0.0,
                    closure_threshold = 0.0, closure_penelty = 0.0)
    
    transition_parameters = vcat(SurplusProduction.reparam(MSY,ratio),rho)
    environmental_shocks_variance = environmental_variance * (1-auto_correlation^2)
    parameters(FMSY,MSY,auto_correlation,transition_parameters)  
end

function transition(state,action,parameters)
    # unpack parameters
    pars = parameters.transition_parameters
    growth_rate = pars[1];density_dependence=pars[2];auto_correlation = pars[3]
    harvest_limit = actions
    
    # transform state variables 
    biomass = exp(state[1])
    environment = exp(state[2])
    
    # compue transitions
    harvest,biomass=convert_harvest(biomass, harvest_limit)
    biomass = growth_rate*biomass/(1+(growth_rate-1)*biomass/density_dependence) 
    biomass *= environment
    if biomass <=0.00001
        biomass = 0.1
    end 
    return [log(biomass),auto_correlation*state[2]]
end 


function reward(state,action,parameters)
    biomass = exp(state[1])
    harvest = action 
    effort = harvest/biomass
    harvest,biomass = SurplusProduction.convert_harvest(biomass, harvest)
    
    revenue = parameters.price*harvest
    costs = parameters.cost_of_effort*effort + parameters.nonlinear_costs*harvest
    rewards = revenue - costs
    if biomass < parameters.over_fished_threshold
        rewards += -1*parameters.over_fished_penelty
    end
    if harvest < parameters.closure_threshold   
        rewards += -1*parameters.closure_penelty 
    end 
    return rewards
end 

const observation_matrix = [1.0 0.0; 0.0 0.0]



end 