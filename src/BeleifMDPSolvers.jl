"""
    Algorithms


Initialize:
- value function
- POMDP object
- Bellman intermidiate

Algorithms:
gridded VFI 
Monte Carlo VFI

Policy iteration 
ADP chain (Policy iteration)
"""
module BeleifMDPSolvers

#using SharedArrays

include("ValueFunctions.jl")
include("BellmanOpperators.jl")


"""
    kalmanFilterSolver

This object stores all of the data required to solve a beleif state MDP 
representing the beleif dynamics with an unscented kalman filter. This includes
a POMDP object that stores the information required to solve the problem a
bellman opperator intermidiate that is used to improve the performance of the
algorith by allowing many operatios to be done inplace, a value funtion 
object, and a policy function object. In addition to these primary objects that 
are used to solve the object also stores data on the performance of the algorith as
strings under the algorithm and warnings 

"""
mutable struct kalmanFilterSolver{T1,T2}
    POMDP::BellmanOpperators.POMDPs.POMDP_KalmanFilter{T1,T2}
    bellmanIntermidiate::AbstractVector{BellmanOpperators.bellmanIntermidiate}
    obsBellmanIntermidiate::AbstractVector{BellmanOpperators.obsBellmanIntermidiate}
    valueFunction::ValueFunctions.adjGausianBeleifsInterp
    policyFunction
    optimizer::String
    algorithm::String
    warnngs::String
end 



mutable struct kalmanFilterSolver1d{T1,T2}
    POMDP::BellmanOpperators.POMDPs.POMDP_KalmanFilter{T1,T2}
    bellmanIntermidiate::AbstractVector{BellmanOpperators.bellmanIntermidiate}
    obsBellmanIntermidiate::AbstractVector{BellmanOpperators.obsBellmanIntermidiate}
    valueFunction::ValueFunctions.guasianBeleifsInterp1d_Bsplines
    policyFunction
    optimizer::String
    algorithm::String
    warnngs::String
end 


"""
    init


The kalman filter POMDP problem is defined with several components:

A state transition function T. For the Kalman filter algorithm this is 
defined with three user inputs, to representatins of the deterministic state 
transition function and a covariance matrix.

T - the transition function defines the expected future state given a state action pair:  T(x,a) = E[x']
T! - inplace verison of the transition function:  T!(x,a) -> x = E[x']
Sigma_N - The covarinace of the process noise (which is assumed to be gausian)


The reward function R. This represents the expected rewards given a state action pair
(x,a) it is defined as a function by the user. 
R - The reward functon maps a state action pair to the within period profits 

The observation model describes the likelihood of an observaiton y given a state aciton pair
(x,a). It is defined by a deterministic observaiton function and a covariance matrix.  
H - The observaiton model:  H(x,a) = E[y]
Sigma_O - the covariance of the observaiton noise (which is assumed to be gausian)

The action space A. This is represented by discrete set of alternatives
a bounded case of  


Sigma_N - The covarinace of the process noise (which is assumed to be gausian)
"""
function initSolver(T!::Function,
            T::Function, 
            R::Function,
            dims_s_quad::Int64,
            R_obs::Function,
            H::Function,
            Sigma_N::AbstractMatrix{Float64},
            Sigma_O::Function,
            delta::Float64,
            actions,
            observations,
            lower_mu,
            upper_mu; 
            m_Quad_x = 10,
            m_Quad_y = 10,
            m_Quad = 10,
            n_grids_obs = 20,
            n_grid = 7)
    
    if length(lower_mu) == 2
        optimizer = "brute force"
        nthread = Threads.nthreads()
        POMDP = BellmanOpperators.POMDPs.init(T!, T, R, R_obs, H,Sigma_N, Sigma_O, delta, actions, observations)
        # set intermidiate
        dims_x = dims_s_quad #size(Sigma_N)[1]
        dims_y = size(Sigma_O(actions[1], observations[1]))[1]
        bellmanIntermidiate = broadcast(i -> BellmanOpperators.init_bellmanIntermidiate(dims_x,dims_s_quad,
                                        dims_y,m_Quad_x,m_Quad_y),1:nthread)
        obsBellmanIntermidiate = broadcast(i ->BellmanOpperators.init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP),1:nthread)
        # default to 30 grid point for observed component and 7 for uncertinaty adjustment
        grids_obs = n_grids_obs
        grids_unc = n_grid
        valueFunction = ValueFunctions.init_adjGausianBeleifsInterp(grids_obs , grids_unc, lower_mu, upper_mu)
        return kalmanFilterSolver{BellmanOpperators.POMDPs.discreteActions,Function}(POMDP,bellmanIntermidiate,obsBellmanIntermidiate,
        "NA",valueFunction, optimizer, "Two stage VFI", "Initialized")
    elseif length(lower_mu) == 1
        optimizer = "brute force"
        nthread = Threads.nthreads()
        POMDP = BellmanOpperators.POMDPs.init(T!, T, R, R_obs, H,Sigma_N, Sigma_O, delta, actions, observations)
        # set intermidiate
        dims_x = dims_s_quad#size(Sigma_N)[1]
        dims_y = size(Sigma_O(actions[1], observations[1]))[1]
        bellmanIntermidiate = broadcast(i -> BellmanOpperators.init_bellmanIntermidiate(dims_x,dims_s_quad,dims_y,
                                        m_Quad_x,m_Quad_y),1:nthread)
        # default to 30 grid point for observed component and 7 for uncertinaty adjustment
        grids_unc = n_grids_obs
        upper_sigma = 0.23
        valueFunction = ValueFunctions.init_guasianBeleifsInterp1d(lower_mu[1],upper_mu[1],upper_sigma,grids_unc)
        return kalmanFilterSolver1d{BellmanOpperators.POMDPs.discreteActions,Function}(POMDP,bellmanIntermidiate,obsBellmanIntermidiate,
        "NA",valueFunction, optimizer, "Two stage VFI", "Initialized")        
    end 
end 

function initSolver(T!::Function,
            T::Function, 
            R::Function,
            dims_s_quad::Int64,
            R_obs::Function,
            H::AbstractMatrix{Float64},
            Sigma_N::AbstractMatrix{Float64},
            Sigma_O::Function,
            delta::Float64,
            actions,
            observations,
            lower_mu,
            upper_mu; 
            m_Quad_x = 10,
            m_Quad_y = 10,
            m_Quad = 10,
            n_grids_obs = 20,
            n_grid = 7)
    if length(lower_mu)  == 2
        optimizer = "brute force"
        POMDP = BellmanOpperators.POMDPs.init(length(lower_mu),T!, T, R, R_obs, H,Sigma_N, Sigma_O, delta, actions, observations)
        # set intermidiate
        dims_x = dims_s_quad #size(Sigma_N)[1]
        dims_y = size(Sigma_O(actions[1], observations[1]))[1]
        bellmanIntermidiate = broadcast(i -> 
                                        BellmanOpperators.init_bellmanIntermidiate(dims_x,dims_s_quad,dims_y,m_Quad_x,m_Quad_y),
                                        1:Threads.nthreads())
        obsBellmanIntermidiate = broadcast(i ->BellmanOpperators.init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP),1:Threads.nthreads())
        # default to 30 grid point for observed component and 7 for uncertinaty adjustment
        grids_obs = n_grids_obs
        grids_unc = n_grid 
        valueFunction = ValueFunctions.init_adjGausianBeleifsInterp(grids_obs , grids_unc, lower_mu, upper_mu)
        policyFuntion = ValueFunctions.init_policyFunctionGaussian(grids_obs , grids_unc, lower_mu, upper_mu,
                                                                    1,1,["action"],["observation"])
        return kalmanFilterSolver{BellmanOpperators.POMDPs.discreteActions,AbstractMatrix{Float64}}(POMDP,bellmanIntermidiate,obsBellmanIntermidiate,valueFunction, 
            policyFuntion,optimizer,  "Two stage VFI", "Initialized")
        
    elseif length(lower_mu) == 1
        optimizer = "brute force"
        POMDP = BellmanOpperators.POMDPs.init(length(lower_mu),T!, T, R, R_obs, H,Sigma_N, Sigma_O, delta, actions, observations)
        # set intermidiate
        dims_x = dims_s_quad#size(Sigma_N)[1]
        dims_y = size(Sigma_O(actions[1], observations[1]))[1]
        bellmanIntermidiate = broadcast(i -> 
                                        BellmanOpperators.init_bellmanIntermidiate(dims_x,dims_s_quad,dims_y,m_Quad_x,m_Quad_y),
                                        1:Threads.nthreads())
        obsBellmanIntermidiate = broadcast(i ->BellmanOpperators.init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP),1:Threads.nthreads())

        grids_unc = n_grids_obs
        upper_sigma = 0.23
        
        #valueFunction = ValueFunctions.init_guasianBeleifsInterp1d(lower_mu[1],upper_mu[1],upper_sigma,grids_unc)
        valueFunction = ValueFunctions.init_guasianBeleifsInterp1d_Bsplines(upper_mu[1],lower_mu[1],2.0,n_grids_obs)
        
        return kalmanFilterSolver1d{BellmanOpperators.POMDPs.discreteActions,AbstractMatrix{Float64}}(POMDP,bellmanIntermidiate,
                obsBellmanIntermidiate,valueFunction, 
                "NA",optimizer,  "Two stage VFI", "Initialized")   
    end 
end 




function initSolver(T!::Function,
            T::Function, 
            R::Function,
            R_obs::Function,
            H::AbstractMatrix{Float64},
            Sigma_N::AbstractMatrix{Float64},
            Sigma_O::Function,
            delta::Float64,
            upper_act::Float64, 
            lower_act::Float64,
            upper_obs::Float64,
            lower_obs::Float64,
            lower_mu,
            upper_mu; 
            m_Quad_x = 10,
            m_Quad_y = 10,
            m_Quad = 10,
            n_grids_obs = 20,
            n_grid = 7)
    optimizer = "passive 1d"
    POMDP = BellmanOpperators.POMDPs.init(T!, T, R, R_obs, H,Sigma_N, Sigma_O, delta, upper_act, lower_act, upper_obs, lower_obs)
    # set intermidiate
    dims_x = size(Sigma_N)[1]
    dims_y = size(Sigma_O([(upper_act+lower_act)/2.0], [(upper_obs+lower_obs)/2.0]))[1]
    bellmanIntermidiate = broadcast(i -> BellmanOpperators.init_bellmanIntermidiate(dims_x,dims_y,m_Quad_x,m_Quad_y),1:Threads.nthreads())
    obsBellmanIntermidiate = broadcast(i ->BellmanOpperators.init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP),1:Threads.nthreads())
    # default to 30 grid point for observed component and 7 for uncertinaty adjustment
    grids_obs = n_grids_obs
    grids_unc = n_grid 
    valueFunction = ValueFunctions.init_adjGausianBeleifsInterp(grids_obs , grids_unc, lower_mu, upper_mu)
    policyFuntion = ValueFunctions.init_policyFunctionGaussian(grids_obs , grids_unc, lower_mu, upper_mu,
                                                                1,1,["action"],["observation"])
    kalmanFilterSolver{BellmanOpperators.POMDPs.boundedActions1d,AbstractMatrix{Float64}}(POMDP,bellmanIntermidiate,obsBellmanIntermidiate,valueFunction, 
        policyFuntion ,optimizer,  "Two stage VFI", "Initialized")
end 

##############################################
### Bellman opperator for observed systems ###
##############################################


"""
    solve_observed(kalmanFilterSolver)

Solves the dynamic program for the fully observed version of the model using
value function iteratation over a set of nodes used in the funciton aproximation. 

Currently this only supports methods that use the adjGausianBeleifsInterp value 
function from the ValueFunctions.jl module. 
"""
function solve_observed!(kalmanFilterSolver;tol=10^-5,max_iter = 3*10^2)
    #tol = 10^-5
    #max_iter = 3*10^2
    test = tol+1.0
    
    nodes = kalmanFilterSolver.valueFunction.baseValue.grid 
    vals = zeros(length(nodes))
        
    vals0 = zeros(length(nodes))
        
    tol *=  length(nodes)
    test *= length(nodes)
    
    iter = 0
    print("here")
    while (test > tol) && (iter < max_iter)
        iter += 1
        print(iter," ")
        print(iter < max_iter, " ")
        println(test)
        i = 0
        # get updated values 
        vals0 .= vals 
        for x in kalmanFilterSolver.valueFunction.baseValue.grid
            i+=1
            #intermidiate = BellmanOpperators.init_obsBellmanIntermidiate(2,3, kalmanFilterSolver.POMDP)
            #print(vcat([x[1],x[2]],zeros(size(kalmanFilterSolver.POMDP.Sigma_N)[1] - 2)))
            vals[i] = BellmanOpperators.obs_Bellman(vcat([x[1],x[2]],zeros(size(kalmanFilterSolver.POMDP.Sigma_N)[1] - 2)),
                            kalmanFilterSolver.obsBellmanIntermidiate[1],  
                            kalmanFilterSolver.valueFunction.baseValue, 
                            kalmanFilterSolver.POMDP, kalmanFilterSolver.optimizer)
        end 
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update_base!(kalmanFilterSolver.valueFunction,vals)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Observed model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: Observed model solved"
    else 
        kalmanFilterSolver.warnngs = "Observed model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: Observed model failed"
    end 
    
end

function solve_observed_parallel!(kalmanFilterSolver;tol=10^-5,max_iter = 3*10^2)
    #tol = 10^-4
    test = tol+1.0
    
    nodes = kalmanFilterSolver.valueFunction.baseValue.grid 
    vals = zeros(length(nodes))
        
    vals0 = zeros(length(nodes))
        
    tol *=  length(nodes)
    test *= length(nodes)
    threadids = zeros(length(nodes))
    iter = 0
    print("here")
    while (test > tol) && (iter < max_iter)
        iter += 1
        print(iter)
        print(" ")
        println(test)
        
        i = 0
        # get updated values 
        vals0 .= vals 
        
        Threads.@threads for i in 1:length(kalmanFilterSolver.valueFunction.baseValue.grid)
            x = kalmanFilterSolver.valueFunction.baseValue.grid[i]
            vals[i] = BellmanOpperators.obs_Bellman(vcat([x[1],x[2]],zeros(size(kalmanFilterSolver.POMDP.Sigma_N)[1] - 2)),
                            kalmanFilterSolver.obsBellmanIntermidiate[Threads.threadid()],
                            kalmanFilterSolver.valueFunction.baseValue, 
                            kalmanFilterSolver.POMDP,kalmanFilterSolver.optimizer)
        end 
        
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update_base!(kalmanFilterSolver.valueFunction,vals)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Observed model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: Observed model solved"
    else 
        kalmanFilterSolver.warnngs = "Observed model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: Observed model failed"
    end 
    
end


function update_policyFunction_obs!(kalmanFilterSolver)

    nodes = kalmanFilterSolver.valueFunction.baseValue.grid 
    
    actionDims = kalmanFilterSolver.policyFunction.actionDims
    observationDims = kalmanFilterSolver.policyFunction.observationDims
    
    action_vals = broadcast(i -> zeros(length(nodes)), 1:actionDims)
    observation_vals = broadcast( i-> zeros(length(nodes)), 1:observationDims) 
    i=0
    for x in kalmanFilterSolver.valueFunction.baseValue.grid
        i+=1
        action = BellmanOpperators.obs_Policy([x[1],x[2]],
                            kalmanFilterSolver.obsBellmanIntermidiate[1],
                            kalmanFilterSolver.valueFunction.baseValue, 
                            kalmanFilterSolver.POMDP,kalmanFilterSolver.optimizer)
        for j in 1:actionDims
            action_vals[j][i] = action[j]
        end 
    
    end 
    
    ValueFunctions.update_policyFunctionGaussian_base!(kalmanFilterSolver.policyFunction, action_vals)
    return action_vals
end 

##############################################
###    Bellman opperator for full model    ###
##############################################


function augmented_nodes(kalmanFilterSolver)
    nodes=kalmanFilterSolver.valueFunction.uncertantyAdjustment.nodes
    n_nodes = length(nodes)
    n_dims_aug = size(kalmanFilterSolver.POMDP.Sigma_N)[1]
    n_dims_obs = 2
    augmented_nodes = broadcast( i-> (zeros(n_dims_aug),zeros(n_dims_aug,n_dims_aug)),1:n_nodes)
    i=0
    for s in nodes
        i+=1
        
        augmented_nodes[i][1][1:2] = s[1]
        augmented_nodes[i][2][1:2,1:2] = s[2]
        augmented_nodes[i][2][(n_dims_obs+1):n_dims_aug,(n_dims_obs+1):n_dims_aug] = 
                    kalmanFilterSolver.POMDP.Sigma_N[(n_dims_obs+1):n_dims_aug,(n_dims_obs+1):n_dims_aug]
    end 
    return augmented_nodes
end 


function augmented_nodes(kalmanFilterSolver::kalmanFilterSolver1d)
    nodes=kalmanFilterSolver.valueFunction.grid #chebyshevInterpolation.
    n_nodes = length(nodes)
    n_dims_aug = size(kalmanFilterSolver.POMDP.Sigma_N)[1]
    n_dims_obs = 1
    augmented_nodes = broadcast( i-> (zeros(n_dims_aug),zeros(n_dims_aug,n_dims_aug)),1:n_nodes)
    i=0
    for s in nodes
        i+=1
        augmented_nodes[i][1][1] = log(s[1]) .- 0.5*s[2]#^2
        augmented_nodes[i][2][1,1] = s[2]
        augmented_nodes[i][2][(n_dims_obs+1):n_dims_aug,(n_dims_obs+1):n_dims_aug] = 
                    kalmanFilterSolver.POMDP.Sigma_N[(n_dims_obs+1):n_dims_aug,(n_dims_obs+1):n_dims_aug]
    end 
    return augmented_nodes
end 

function solve(kalmanFilterSolver; max_iter = 2*10^2,tol = 10^-3)
   
    test = tol+1.0
    nodes = augmented_nodes(kalmanFilterSolver) 
    
    vals = zeros(length(nodes)) 
    vals0 = zeros(length(nodes))
    tol *=  length(nodes)
    test *= length(nodes)
    iter = 0
    
    while (test > tol) && (iter < max_iter)
        print(iter)
        print(" ")
        println(test)

        iter += 1
        i = 0
        # get updated values 
        vals0 .= vals
        for s in nodes
            i+=1
            if mod(i, 100) == 0
                print(i/length(nodes))
                print(" ")
            end 

            vals[i] = BellmanOpperators.Bellman!(s, kalmanFilterSolver.bellmanIntermidiate[1], #Threads.theadid()
                                        kalmanFilterSolver.valueFunction,kalmanFilterSolver.POMDP, 
                                        kalmanFilterSolver.optimizer)
        end 
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update!(kalmanFilterSolver.valueFunction,vals)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Full model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: full model solved"
    else 
        kalmanFilterSolver.warnngs = "Full model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: full model failed"
    end 
end 




function solve_parallel!(kalmanFilterSolver;max_iter = 150,tol = 10^-3)
    test = tol+1.0
    
    nodes = augmented_nodes(kalmanFilterSolver) 
    vals = zeros(length(nodes)) 
    vals0 = zeros(length(nodes))
    tol *=  length(nodes)
    test *= length(nodes)
    iter = 0
    
    while (test > tol) && (iter < max_iter)
        print(iter)
        print(" ")
        println(test)
        
        iter += 1
        i = 0
        # get updated values 
        vals0 .= vals
        
        Threads.@threads for i in 1:length(nodes)
            
            s = nodes[i]
            #println(s)
            vals[i] = BellmanOpperators.Bellman!(s,
                                        kalmanFilterSolver.bellmanIntermidiate[Threads.threadid()],
                                        kalmanFilterSolver.valueFunction,
                                        kalmanFilterSolver.POMDP, 
                                        kalmanFilterSolver.optimizer)
        end 
        # update value function 
        test = sum((vals .- vals0).^2)
        ValueFunctions.update!(kalmanFilterSolver.valueFunction,vals)
    end 
    
    if iter < max_iter
        kalmanFilterSolver.warnngs = "Full model converged"
        kalmanFilterSolver.algorithm = "two stage VFI: full model solved"
    else 
        kalmanFilterSolver.warnngs = "Full model failed"
        kalmanFilterSolver.algorithm = "Two stage VFI: full model failed"
    end 
end 


function update_policyFunction!(kalmanFilterSolver)
    nodes = kalmanFilterSolver.valueFunction.uncertantyAdjustment.nodes
    
    actionDims = kalmanFilterSolver.policyFunction.actionDims
    observationDims = kalmanFilterSolver.policyFunction.observationDims
    
    action_vals = broadcast(i -> zeros(length(nodes)), 1:actionDims)
    observation_vals = broadcast( i-> zeros(length(nodes)), 1:observationDims) 
    
    Threads.@threads for i in 1:length(nodes)
        s = nodes[i]
        action,observation = BellmanOpperators.Policy!(s,
                            kalmanFilterSolver.bellmanIntermidiate[Threads.threadid()],
                            kalmanFilterSolver.valueFunction, 
                            kalmanFilterSolver.POMDP,kalmanFilterSolver.optimizer)
        for j in 1:actionDims
            action_vals[j][i] = action[j]
        end 
        for j in 1:observationDims
            observation_vals[j][i] = observation[j]
        end 
    end 
    
    ValueFunctions.update_policyFunctionGaussian_adjustment!(kalmanFilterSolver.policyFunction, action_vals, observation_vals)
    
end 



end # module 