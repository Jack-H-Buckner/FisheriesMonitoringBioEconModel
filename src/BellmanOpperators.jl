"""
    BellmanOpperators

Provides methods to evaluate the Bellman equation for each of the POMDP objects. 

The primay function defined in this module is `bellman` which evaluates the bellman opperator give
a POMDP object, and initial state and a value function object. If an action variable `a` is provided 
then the algorithm will calcualte the expected value of taking that action in the current state. If
no action is provided then the function will solve for the best action using the `Optim.jl` package. 

There are two primary computational challenges related to solving the bellman opperator: optimization and 
integration. Seperate methos are defined to apply the most aproporate method depending on the input arguments.

For `POMDP_particleFilter` objects Montecarlo integration is used, and Gauss Hermite quadrature is used for 
`POMDP_kalmanFilter` objects. The implementation of the Gausian quadrature depends on weather or not the observation
model is linear. If it is linear then 

The optimizaiton method is chosen based on the Type of the POMDP.actions object. Nelder Mead is used for 
multivariate continuous aciton, Golden search for univariate continuous, and brute force is used for 
discrete action spaces. 
"""
module BellmanOpperators

using KalmanFilters
using LinearAlgebra
#using Optim

include("MvGaussHermite.jl")
include("POMDPs.jl")
#####################################
###  Gaussain quadrature and UKF  ###
#####################################

"""
    bellmanIntermidiate

Note that this intermidiate is a work in progress I will add to it to make more
of the operations in place to further optimize the code. 

This structure provides a place to allocate memory for all of the values that need
to be computed each time the bellman opporator is coputed. This allows more of the 
opperations to be done in place which should increase performance. 

"""
mutable struct bellmanIntermidiate
    dims_x_quad::Int64 # the number of dimensions of the stat space used to evaluate the expected rewards. 
    Quad_x::MvGaussHermite.mutableQuadrature
    Quad_y::MvGaussHermite.mutableQuadrature
    
    x_hat::AbstractVector{Float64}
    x_cov::AbstractMatrix{Float64}
    y_hat::AbstractVector{Float64}
    y_cov::AbstractMatrix{Float64}
    
    new_states_mat::AbstractVector{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
    new_states_vec::AbstractVector{AbstractVector{Float64}}
    vals::AbstractVector{Float64}
    z::AbstractVector{Float64}
end 




function init_bellmanIntermidiate(dims_s,dims_x_quad, dims_y, m_Quad_x, m_Quad_y)
    x_hat = zeros(dims_s)
    x_cov = 1.0*Matrix(I,dims_s,dims_s)
    y_hat = zeros(dims_y)
    y_cov = 1.0*Matrix(I,dims_y,dims_y)
    
    Quad_x = MvGaussHermite.init_mutable(m_Quad_x,x_hat[1:dims_x_quad],x_cov[1:dims_x_quad,1:dims_x_quad])
    Quad_y = MvGaussHermite.init_mutable(m_Quad_y,y_hat,y_cov)
    
    
    vals = zeros(Quad_y.n)
    new_states_mat = []
    new_states_vec = []
    for i in 1:Quad_y.n
        push!(new_states_mat, (zeros(dims_s), 1.0*Matrix(I,dims_s,dims_s)))
        push!(new_states_vec, zeros(floor(Int,dims_s + dims_s*(dims_s+1)/2)))
    end 
    
    z = 1.0*zeros(round(Int,dims_y  + dims_y * (dims_y+1) /2)) # need to write this in a more general way
    
    return bellmanIntermidiate(dims_x_quad,Quad_x,Quad_y,x_hat,x_cov,y_hat,y_cov,new_states_mat,new_states_vec,vals, z)
    
end 




"""
    propogate_observation_model(x_hat,Cov,H)

propogates state uncertianty through observation model if H is a function
the gassian quadrature is used if it is a matrix then a linear transform
is used. 

x_hat - mean estimate for x
Cov - covariance of beleif state
H - observation model, matrx of function 
"""
function propogate_observation_model(x_hat::AbstractVector{Float64},x_cov::AbstractMatrix{Float64},
                                     H::Function,Quad)
    MvGaussHermite.update!(Quad, x_hat, x_cov)
    vals = H.(Quad.nodes)
    y_hat = sum(vals.*Quad.weights)
    y_cov = sum(broadcast(v-> (v.-y_hat)*transpose(v.-y_hat), vals).* Quad.weights)
    return y_hat, y_cov
end 


function propogate_observation_model(x_hat::AbstractVector{Float64},x_cov::AbstractMatrix{Float64},
                                     H::AbstractMatrix{Float64})
    y_hat = H * x_hat
    y_cov = H * x_cov * transpose(H) 
    return y_hat, y_cov
end 



"""
    new_state(y, x_hat, x_cov,H,Sigma_O)

computes measurement update using KalmanFilters.jl 
"""
function new_state(y::AbstractVector{Float64}, x_hat::AbstractVector{Float64},
                    x_cov::AbstractMatrix{Float64},H::Function,
                    Sigma_O::AbstractMatrix{Float64}, n_dims)
    
    mu = KalmanFilters.measurement_update(x_hat, x_cov,y,H,Sigma_O)
    x_hat1, x_cov1 = KalmanFilters.get_state(mu), KalmanFilters.get_covariance(mu)
    
    return x_hat1[1:n_dims], x_cov1[1:n_dims,1:n_dims]
end 


function new_state(y::AbstractVector{Float64}, x_hat::AbstractVector{Float64},
                    x_cov::AbstractMatrix{Float64},H::AbstractMatrix{Float64},
                    Sigma_O::AbstractMatrix{Float64}, n_dims)
    
    mu = KalmanFilters.measurement_update(x_hat, x_cov,y,H,Sigma_O)
    x_hat1, x_cov1 = KalmanFilters.get_state(mu), KalmanFilters.get_covariance(mu)
    
    return x_hat1[1:n_dims], x_cov1[1:n_dims,1:n_dims]
end 









"""
    reshape_state(x_hat, x_cov)

reshapes the beleif state represented by a mean vector and covariance matrix to a 
vector of length n + n(n+1)/2
"""
function reshape_state(x_hat::AbstractVector{Float64}, x_cov::AbstractMatrix{Float64})
    n = length(x_hat)
    v = zeros(floor(Int,n + n*(n+1)/2))
    v[1:n] .= x_hat
    k = n
    for i in 1:n
        for j in 1:n
            if j <= i
                k += 1
                v[k] = x_cov[i,j]
            end 
        end
    end 
    return v
end 


function reshape_state(B::AbstractVector{Float64})
    d = floor(Int, -3/2+sqrt(9/4+2*length(B)))
    x_hat = B[1:d]
    x_cov = zeros(d,d)
    k = d
    for i in 1:d
        for j in 1:d
            if j == i
                k += 1
                x_cov[i,j] = B[k] 
            elseif j < i
                k += 1
                x_cov[i,j],x_cov[j,i] = B[k],B[k]  
            end 
        end
    end 
    return x_hat, x_cov
end 









"""
    value_expectation!(s,a,POMDP,Quad_X, Quad_y)

data - inplace object, update x_hat and x_cov to evaluate at a given grid point 
a - action
POMDP - problem
V - value function 

"""
function value_expectation!(s, a, obs, data, POMDP, V) 
    
    # I need to add functionality here to run the time update with the augmented filter
    #
    # idea: change the data object to store the full augmented state 
    #    1) update the data intermidiate object to store the augemted mean and covariance matrix 
    #    2) update the new_state funtion to only retun the state variables. 
    #
    # progress:
    #    2) I think I added a piece required to do this 
    #    3) need to update how s is generated 
    #    4) per 3 I will need to updated the value function object to augment the nodes  
    
    # convert state vector to mean and covariance 
    
    data.x_hat, data.x_cov = s
    tu = KalmanFilters.time_update(data.x_hat, data.x_cov, x ->POMDP.T(x,a),  POMDP.Sigma_N)
    data.x_hat, data.x_cov = KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
    
    #observaton quadrature nodes 
    data.y_hat, data.y_cov = propogate_observation_model(data.x_hat,data.x_cov,POMDP.H) # allocation 
    data.y_cov .+= POMDP.Sigma_O(a,obs)
    MvGaussHermite.update!(data.Quad_y,data.y_hat,data.y_cov)
    
     ### new states 
    
    #new_state!(data,H, POMDP.Sigma_O)
    data.new_states_mat = broadcast(y -> new_state(y, data.x_hat,data.x_cov, POMDP.H,
                                    POMDP.Sigma_O(a,obs),POMDP.n_dims), data.Quad_y.nodes)    
    # large allocation 
    #data.new_states_vec = broadcast(x -> reshape_state(x[1],x[2]),data.new_states_mat)
    data.vals = broadcast(x -> V(data.z,x), data.new_states_mat)
    
    return sum(data.vals .* data.Quad_y.weights)
end 





"""
    value_expectation_passive!(s, a, data, POMDP, V)
"""
function value_expectation_passive!(s, a, data, POMDP, V) #::POMDPs.POMDP_KalmanFilter{AbstractMatrix{Float64}}
    
    # convert state vector to mean and covariance 
    data.x_hat, data.x_cov = s
    tu = KalmanFilters.time_update(data.x_hat, data.x_cov, x ->POMDP.T(x,a),  POMDP.Sigma_N)
    data.x_hat, data.x_cov = KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
    
    return V(data.z,(data.x_hat, data.x_cov))
    
end 


"""
    reward_expectation(s,a,POMDP,Quad_X;Quad_epsilon)

Takes expectation over reward function to account for state uncertianty and stochasticity.    
The input Quad_epsilon is optional and should be omitted if POMDP.R takes only two arguments

s - state vector length n + n*(n+1)/2
a - action taken
POMDP - POMDP_KalmandFilter 
Quad_x - integrates over uncertianty in x, will be updated for new vals of x_hat and x_cov 
Quad_epsilon - integrates over variance in rewards 
"""
function reward_expectation!(s, a, obs, data, POMDP)
    data.x_hat, data.x_cov = s
    MvGaussHermite.update!(data.Quad_x,data.x_hat[1:data.dims_x_quad],data.x_cov[1:data.dims_x_quad,1:data.dims_x_quad]) 
    v = sum(broadcast(x -> POMDP.R(x, a,obs),data.Quad_x.nodes) .* data.Quad_x.weights) # large allocation 
    return v
end


function reward_expectation_passive!(s, a, data, POMDP)
    data.x_hat, data.x_cov = s
    MvGaussHermite.update!(data.Quad_x,data.x_hat,data.x_cov) 
    v = sum(broadcast(x -> POMDP.R_obs(x, a),data.Quad_x.nodes) .* data.Quad_x.weights) # large allocation 
    return v
end

"""
    expectation(data, s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            V::Function, POMDP)

Computes the expected value of the bellman opperator given an action 
"""
function expectation!(s, a, obs,  data, POMDP, V)
    EV = value_expectation!(s, a, obs, data, POMDP, V) #value_expectation(s, a,V, POMDP, Quad_x, Quad_y)
    ER = reward_expectation!(s, a, obs, data, POMDP)
    return ER + POMDP.delta*EV
end

    
function expectation_passive!(s, a,  data, POMDP, V)
    EV = value_expectation_passive!(s, a, data, POMDP, V) #value_expectation(s, a,V, POMDP, Quad_x, Quad_y)
    ER = reward_expectation_passive!(s, a, data, POMDP)
    return ER + POMDP.delta*EV
end

    
    
#################################
## Bellman operators, policies ##
#################################


"""
    action_passive!(s, data,POMDP, V)

Identifies the best action choice, not accounting for the measurment update. solve for maximum with
brent search 
"""

function action_passive!(s,data,POMDP,V, optimizer)
    if optimizer == "passive 1d" 
        sol = Optim.optimize(a -> -1*expectation_passive!(s, a,data, POMDP,V), 
                            POMDP.actions.upper_act, POMDP.actions.lower_act, GoldenSection())     
        return sol.minimizer
    elseif optimizer == "passive brute force"
        results = broadcast(a -> expectation_passive!(s,a,data, POMDP,V),POMDP.actions.actions) 
        ind = argmax(results)
        return POMDP.actions.actions[ind]
    end 
end 


     

"""
    Bellman(s,V, POMDP, Quad_x, Quad_y, Quad_epsilon)

Evaluates the bellman opperator, when the action space is discrete by evaluating all choice
and chooseing the best performer.

If an action is provided then is searches for the best observaiton variable given that action
and returns the optimum value. 
"""
function Bellman!(s, data,V, POMDP, optimizer)    
    if optimizer == "brute force"
        results = broadcast(a -> expectation!(s, a[1],a[2],data, POMDP,V),POMDP.actions.choices) 
        ind = argmax(results)
        return results[ind]
    elseif optimizer == "passive 1d"
        a = action_passive!(s, data,POMDP,V,optimizer)
        sol = Optim.optimize(obs -> -1*expectation!(s,a,obs,data, POMDP,V), 
                        POMDP.actions.upper_obs, POMDP.actions.lower_obs,  GoldenSection())  
        return -1.0*sol.minimum
    elseif optimizer == "passive brute force"
        a = action_passive!(s, data,POMDP,V,optimizer)
        results = broadcast(obs -> expectation!(s, a, obs,data, POMDP,V),POMDP.actions.observations) 
        ind = argmax(results)
        return results[ind]
    end
end 
    
    
function Policy!(s, data,V, POMDP, optimizer)    
    if optimizer == "brute force"
        results = broadcast(a -> expectation!(s, a[1],a[2],data, POMDP,V),POMDP.actions.choices) 
        ind = argmax(results)
        return POMDP.actions.choices[ind]
    elseif optimizer == "passive 1d"
        a = action_passive!(s, data,POMDP,V)
        sol = Optim.optimize(obs -> -1*expectation!(s,a,obs,data, POMDP,V), 
                        POMDP.actions.upper_obs, POMDP.actions.lower_obs,  GoldenSection())  
        return a, sol.minimizer
    elseif "passive brute force"
        a = action_passive!(s, data,POMDP,V,optimizer)
        results = broadcast(obs -> expectation!(s, a, obs,data, POMDP,V),POMDP.actions.observations) 
        ind = argmax(results)
        return a, POMDP.actions.observations[ind]
    end
end

##############################################
### Bellman opperator for observed systems ###
##############################################


"""
    obsBellmanIntermidiate

stores data useful for bellman opperator when the
system is fully observed
"""
mutable struct obsBellmanIntermidiate
    x_prime::AbstractVector{Float64}
    Quad::MvGaussHermite.mutableQuadrature
    values::AbstractVector{Float64}
end 


function init_obsBellmanIntermidiate(dims_x,m_Quad, POMDP)
    x = zeros(dims_x)
    Quad =  MvGaussHermite.init_mutable(m_Quad,x,POMDP.Sigma_N)
    values = zeros(Quad.n)
    return obsBellmanIntermidiate(x,Quad,values)
end 

"""
    obs_value_expectation(intermidiate,V,POMDP)

calcualtes the bellman operator given the an action a updates the 
value v and the intermidate
"""
function obs_expectation(x,a,intermidiate,V,POMDP)
    v = 0 # can try to make this in place later
    #print(x)
    intermidiate.x_prime = POMDP.T(x,a)#[1:POMDP.n_dims]
    MvGaussHermite.update!(intermidiate.Quad, intermidiate.x_prime, POMDP.Sigma_N)
    intermidiate.values .= broadcast(x -> V(x[1:POMDP.n_dims]), intermidiate.Quad.nodes)
    v = sum(intermidiate.values .* intermidiate.Quad.weights)
    v *= POMDP.delta
    v += POMDP.R_obs(x,a)
    return v
end 


"""
    Bellman(intermidiate,)

Evaluates the bellman opperator, when the action space is discrete by evaluating all choice
and chooseing the best performer
""" #::POMDPs.POMDP_KalmanFilter{POMDPs.discreteActions,AbstractArray{Float64,2}}
function obs_Bellman(x, intermidiate,V, POMDP, optimizer)
    if (optimizer == "brute force") | (optimizer == "passive brute force")
        results = broadcast(a -> obs_expectation(x,a,intermidiate,V,POMDP),POMDP.actions.actions)
        ind = argmax(results)
        return results[ind]
    elseif optimizer == "passive 1d"
        sol = Optim.optimize(a -> -1*obs_expectation(x,a,intermidiate,V,POMDP), 
                         POMDP.actions.upper_act, POMDP.actions.lower_act, GoldenSection()) 
        return -1.0*sol.minimum#, sol.minimizer
    end
end 


"""
    Bellman(intermidiate,)

Evaluates the bellman opperator, when the action space is discrete by evaluating all choice
and chooseing the best performer
"""
function obs_Policy(x, intermidiate,V, POMDP, optimizer)
    if optimizer == "brute force"
        results = broadcast(a -> obs_expectation(x,a,intermidiate,V,POMDP),POMDP.actions.actions)
        ind = argmax(results)
        return POMDP.actions.actions[ind]
    elseif optimizer == "passive 1d"
        sol = Optim.optimize(a -> -1*obs_expectation(x,a,intermidiate,V,POMDP), 
                         POMDP.actions.upper_act, POMDP.actions.lower_act, Brent()) 
        return sol.minimizer
    end
end



#######################
### other functions ###
#######################


function pf_ukf_value_expectation(s, a, obs, POMDP, V, N1, N2)
    
    # convert state vector to mean and covariance 
    x_hat, x_cov = reshape_state(s) 
    d = Distributions.MvNormal(x_hat, x_cov)
    
    particles = ParticleFilters.init(N1,length(x_hat))

    particles.samples = broadcast(i -> reshape(rand(d,1),2), 1:N1)
    
    ParticleFilters.time_update!(particles,POMDP.T_sim!,a)
    
    x_hat = sum(particles.samples .* particles.weights)
    x_cov = sum(broadcast( i-> (x_hat .- particles.samples[i]) * transpose(x_hat .- particles.samples[i]), 1:N1).* particles.weights)
    
    #observaton quadrature nodes 
    H = x -> POMDP.H(x,a)
    inds = sample(1:N1, N2)
    x = particles.samples[inds]
    y = POMDP.G_sim.(x,a)
    
    newstates = broadcast(i -> zeros(length(s)),1:N2)
    i = 0

    for yt in y
        #print(yt)
        i += 1
        x_hat_t, x_cov_t = new_state(reshape(yt,1), x_hat,x_cov,H,POMDP.Sigma_O(a,obs))
        #x_hat_t, x_cov_t = KalmanFilters.get_state(mu), KalmanFilters.get_covariance(mu)
        newstates[i] = reshape_state(x_hat_t, x_cov_t) 
    end
    
    vals = broadcast(x -> V(x, a), newstates)
    
    return sum(vals)/N2
end 


function pf_ukf_expectation(s, a, obs, POMDP, V, N1, N2)
    # convert state vector to mean and covariance 
    ER = MC_reward_expectation(s,a,POMDP,N2)
    EV = pf_ukf_value_expectation(s, a, POMDP, V, N1, N2)
    return ER + POMDP.delta*EV
end 




function expectation(s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            V::Function, POMDP, Quad_x, Quad_y)
    EV = value_expectation(s, a,V, POMDP, Quad_x, Quad_y)
    ER = reward_expectation(s, a, POMDP, Quad_x)
    return ER + POMDP.delta*EV
end 


function expectation!(data, s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            V::Function, POMDP, Quad_epsilon)
    data.s = s
    EV = value_expectation!(data, a, POMDP, V) #value_expectation(s, a,V, POMDP, Quad_x, Quad_y)
    ER = reward_expectation(s, a, POMDP, data.Quad_x, Quad_epsilon)
    return ER + POMDP.delta*EV
end 


function expectation(s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            V::Function, POMDP, Quad_x, Quad_y, Quad_epsilon)
    EV = value_expectation(s, a,V, POMDP, Quad_x, Quad_y)
    ER = reward_expectation(s, a, POMDP, Quad_x, Quad_epsilon)
    return ER + POMDP.delta*EV
end 


function reward_expectation(s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            POMDP, Quad_x, Quad_epsilon)
    
    # convert state vector to mean and covariance 
    x_hat, x_cov = reshape_state(s) # allocation 
    MvGaussHermite.update!(Quad_x,x_hat,x_cov)
    f = x -> sum(broadcast(eps -> POMDP.R(x, a, eps),Quad_epsilon.nodes) .* Quad_epsilon.weights)
    v = sum(broadcast(x -> f(x) ,Quad_x.nodes) .* Quad_x.weights) # large allocation 
    return v
end


function Bellman(s::AbstractVector{Float64},V::Function, a0, POMDP, Quad_x, Quad_y)
    results = Optim.optimize(a -> -1*expectation(s, a,V, POMDP, Quad_x, Quad_y), a0, NelderMead())
    return results.minimum
end 


function Policy(s::AbstractVector{Float64},V::Function, a0, POMDP, Quad_x, Quad_y)
    results = Optim.optimize(a -> -1*expectation(s, a,V, POMDP, Quad_x, Quad_y), a0, NelderMead())
    return results.minimizer
end 



function value_expectation(s::AbstractVector{Float64}, a::AbstractVector{Float64}, 
                            V::Function, POMDP, Quad_x, Quad_y)
    
    # convert state vector to mean and covariance 
    x_hat, x_cov = reshape_state(s) # allocation 
    
    ### time update 
    tu = KalmanFilters.time_update(x_hat,x_cov, x ->POMDP.T(x,a),  POMDP.Sigma_N)
    x_hat, x_cov = KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
    
    #observaton quadrature nodes 
    H = x -> POMDP.H(x,a) # define measurment function # allocation 
    y_hat, y_cov = propogate_observation_model(x_hat, x_cov,H,Quad_x) # allocation 
    y_cov .+= POMDP.Sigma_O(a, obs)
    MvGaussHermite.update!(Quad_y,y_hat,y_cov)
    
    ### new states 
    new_states_mat = broadcast(y -> new_state(y, x_hat,x_cov, H, POMDP.Sigma_O(a,obs)), Quad_y.nodes) # large allocation 
    new_states_vec = broadcast(x -> reshape_state(x[1],x[2]),new_states_mat)
    
    vals = broadcast(x -> V(x, a), new_states_vec) # large allocation 
    return sum(vals .* Quad_y.weights)
end 


#########################################
###  Monte Carlo and particle filter  ###
#########################################


# include("ParticleFilters.jl")
# using Distributions


# """
#     MC_value_expectation!(s,a,POMDP,Quad_X, Quad_y)
# """
# function MC_value_expectation(s, a, obs, POMDP, V, N1, N2)
    
#     # convert state vector to mean and covariance 
#     x_hat, x_cov = s
#     d = Distributions.MvNormal(x_hat, x_cov)
    
#     particles = ParticleFilters.init(N1,length(x_hat))

#     particles.samples = broadcast(i -> reshape(rand(d,1),2), 1:N1)
    
#     ParticleFilters.time_update!(particles,POMDP.T_sim!,a)
    
#     #observaton quadrature nodes 
#     inds = sample(1:N1, N2)
#     x = particles.samples[inds]
#     y = broadcast(x->POMDP.G_sim(x,a,obs),x)
    
#     newstates = broadcast(i -> zeros(length(s)),1:N2)
#     i = 0
#     for yt in y
#         i += 1
#         samples, weights = ParticleFilters.bayes_update(particles,POMDP.G,yt,a,obs)
#         x_hat = sum(samples .* weights)
#         x_cov = sum(broadcast( i-> (x_hat .- samples[i]) * transpose(x_hat .- samples[i]), 1:N1).* weights)
#         newstates[i] = reshape_state(x_hat, x_cov) 
#     end
    
#     vals = broadcast(x -> V(x, a), newstates)
    
#     return sum(vals)/N2
# end 




# function MC_reward_expectation(s,a,obs, POMDP,N)
#     # convert state vector to mean and covariance 
#     x_hat, x_cov = s
#     d = Distributions.MvNormal(x_hat, x_cov)
#     samples = broadcast(i -> reshape(rand(d,1),2), 1:N)
#     vals = broadcast( x-> POMDP.R(x,a,obs),samples)
#     return sum(vals)/N
# end


# function MC_expectation(s,a,obs,POMDP,V,N1,N2)
#     # convert state vector to mean and covariance 
#     ER = MC_reward_expectation(s,a,obs,POMDP,N2)
#     EV = MC_value_expectation(s, a,obs,POMDP, V, N1, N2)
#     return ER + POMDP.delta*EV
# end

end # module 