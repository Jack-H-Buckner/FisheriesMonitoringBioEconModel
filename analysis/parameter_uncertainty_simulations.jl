using KalmanFilters
using LinearAlgebra
using Distributions

include("SurplusProduction.jl")

growth(biomass,r,b) = r*biomass/(1+b*biomass)
BMSY(r,b) = 2*b*(sqrt(r)-1)/(2*b^2) 
MSY(r,b) = growth(BMSY(r,b),r,b) - BMSY(r,b)
FMSY(r) = sqrt(r)-1
growth_rate(_BMSY,_FMSY) = (_FMSY+1)^2
density_dependence(_BMSY,_FMSY) = _FMSY/_BMSY
MSY2(_FMSY,_BMSY) = MSY(growth_rate(_BMSY,_FMSY),density_dependence(_BMSY,_FMSY))
    

"""
"""
function deterministic_transition(state,action)
    # unpack state
    biomass=exp(state[1]);_growth_rate=exp(state[2]);_BMSY=exp(state[3])
    _FMSY = FMSY(_growth_rate)
    _density_dependence = density_dependence(_BMSY,_FMSY)
    
    # unpack action
    harvest=action
    
    # update biomass

    harvest,biomass = SurplusProduction.convert_harvest(biomass, harvest)

    biomass = growth(biomass,_growth_rate,_density_dependence)
    
    return [log(biomass),state[2],state[3],state[4]]
end 


observation_model = [1 0 0 1]

function compute_inital_beleif_state(_FMSY,_BMSY, 
                                    processes_noise_covariance,
                                    observation_noise_covaraince, 
                                    harvest_time_series; 
                                    initail_beleifs_CV = 1.0,
                                    observation_model = [1 0 0 1])
    
    _MSY = MSY2(_FMSY,_BMSY)
    _growth_rate = growth_rate(_BMSY,_FMSY)
    _density_dependence = density_dependence(_BMSY,_FMSY)
    biomass0 =  (_growth_rate - 1)/_density_dependence
    println("biomass t=0: ", biomass0 )
    inital_beleifs_covariance = log(initail_beleifs_CV^2+1)*Matrix(I,4,4)
    inital_beleifs_mean = log.([biomass0,_growth_rate,_BMSY,1.0])
    inital_state = log.([biomass0,_growth_rate,_BMSY,1.0])
    
    ## first time step
    # state update
    state = deterministic_transition(inital_state,harvest_time_series[1])
    state .+= rand(Distributions.MvNormal(zeros(4),processes_noise_covariance))
    
    # beleif update
    observation = observation_model * state 
    observation .+= rand(Distributions.MvNormal([0.0],observation_noise_covaraince))
    beleif_state = time_update(inital_beleifs_mean, inital_beleifs_covariance, 
                                x ->deterministic_transition(x,harvest_time_series[1]), 
                                processes_noise_covariance)
    beleif_state = measurement_update(get_state(beleif_state), get_covariance(beleif_state),
                                     observation, observation_model, observation_noise_covaraince)
    
    
    T = length(harvest_time_series)
    for t in 2:T
        # state update
        state = deterministic_transition(state ,harvest_time_series[1])
        state .+= rand(Distributions.MvNormal(zeros(4),processes_noise_covariance))

        # beleif update
        observation = observation_model * state 
        observation .+= rand(Distributions.MvNormal([0.0],observation_noise_covaraince))
        beleif_state = time_update(get_state(beleif_state), get_covariance(beleif_state), 
                                    x ->deterministic_transition(x,harvest_time_series[1]), 
                                    processes_noise_covariance)
        beleif_state = measurement_update(get_state(beleif_state), get_covariance(beleif_state),
                                         observation, observation_model, observation_noise_covaraince)

    end 
    
    return get_state(beleif_state), get_covariance(beleif_state), state
    
end


function update_beleif_state(state_estimate,state_covaraince, model)
    beleif_state = time_update(state_estimate, state_covaraince, 
                                x ->deterministic_transition(x,harvest_time_series[1]), 
                                model.processes_noise_covariance)
    return get_state(beleif_state), get_covariance(beleif_state)
end 


function assessment(state_estimate,state_covaraince,harvest_series,observation_series)
    
end 


function main()
    harvest_time_series = vcat(vcat(collect(0.0:0.5:15.0),collect(15.0:-1.0:5.0)),repeat([5.0],10))
    
    harvest_time_series = harvest_time_series[1:40]
    
    _FMSY = 0.2
    _BMSY = 50.0  
    processes_noise_covariance = [0.1 0.0 0.0 0.0;
                                  0.0 0.0001 0.0 0.0;
                                  0.0 0.0 0.0001 0.0;
                                  0.0 0.0 0.0 0.0001;]
    observation_noise_covaraince = [0.1;;]
    
    state_estimate,state_covaraince,state = compute_inital_beleif_state(_FMSY,_BMSY, 
                                    processes_noise_covariance,
                                    observation_noise_covaraince, 
                                    harvest_time_series)
    
    println("true values: ", exp.(state))
    println("estimates: ", exp.(state_estimate))
    println("errors: ",exp.(sqrt.(diag(state_covaraince)).-1))
    S = state_covaraince
    D = Matrix(I,4,4).*sqrt.(diag(S))
    println("correlations: \n", inv(D)*S*inv(D)[1,:])
    println(inv(D)*S*inv(D)[2,:])
    println(inv(D)*S*inv(D)[3,:])
    println(inv(D)*S*inv(D)[4,:])
    println("total uncertianty: ", det(S[1:3,1:3]))
        
end 

main()