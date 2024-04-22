include("../src/MOMDPs.jl")
using LinearAlgebra


function f(beleifState)
    det(beleifState.covariance_matrix) + prod(beleifState.estimated_states) + prod(beleifState.known_states)
end 


function test_value_function()
    # initianize a value function with arbitraty bounds
    aproximation_order = [3,3,3,3,3,3]
    mean_bounds = [0.01 0.01;1.0 1.0]
    variance_bounds = [0.01 0.01; 1.0 1.0]
    max_correlation = 0.9
    known_states_bounds = [-1.0;1.0;;]
    V=MOMDPs.init_ValueFunction(aproximation_order,mean_bounds,variance_bounds,max_correlation,known_states_bounds;
                            n_nodes_max = 10^5)
    
    
    # set values of value function
    V.values = f.(V.nodes)
    MOMDPs.update!(V)
    
    # test vaalues at knots
    ind = argmax(broadcast(ind->V(zeros(6),V.nodes[ind]) .- V.values[ind], 1:length(V.values)).^2)
    max_error = broadcast(ind->V(zeros(6),V.nodes[ind]) .- V.values[ind], 1:length(V.values))[ind]
    
    @assert max_error < 10^-5
    
    # test of grid 
    beleifState = MOMDPs.init_BeleifState([0.32, 0.015],[0.14 0.015; 0.015 0.12],[-0.13])
    true_val = f(beleifState)
    itp_val = V(zeros(6),beleifState)
    error = (true_val - itp_val)^2
    @assert error < 10^-2
end 

function main()
    test_value_function()
end 
main()
