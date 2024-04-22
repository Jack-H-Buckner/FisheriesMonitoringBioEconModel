module MOMDPs
using Interpolations
using LinearAlgebra
function cov_to_cor(covariance_matrix)
    inv(sqrt.(Diagonal(covariance_matrix)))*covariance_matrix*inv(sqrt.(Diagonal(covariance_matrix)))
end 

#### quadrature methods
include("MVGaussHermite.jl")

## value functions

"""
beleif_state

stores data to define a decision makers beleif state

mean - Vector: expected values of partially observed variables
Cov - Matrix: Covariance matrix of partially observed variables 
observed - Vector: value of observed variables 
"""
mutable struct BeleifState
    estimated_states::AbstractVector{Float64}
    covariance_matrix::AbstractMatrix{Float64}
    correlation_matrix::AbstractMatrix{Float64}
    known_states::AbstractVector{Float64} 
end 

function init_BeleifState(mean,covariance_matrix,known_states) 
    correlation_matrix = zeros(size(covariance_matrix))
    return BeleifState(mean,covariance_matrix,correlation_matrix,known_states)
end 

"""
scaling

stores data required to scale beleif state variables to the range of the value function 

m - vector with number of grid point in each dimension of the interpoliation object
mean_bounds - 2 by d matrix, top row has lower bounds bottom row has upper bounds for the mean value of the partially observed variables
variance_bounds - 2 by d matrix, top row has lower bounds bottom row has upper bounds for the variance of the partially observed variables
max_correlation - float giving an upper bound for the covariance 
known_states_bounds - 2 by d matrix, top row has lower bounds bottom row has upper bounds for the value of the observed variables
"""
struct ValueFunctionBoundary
    aproximation_order::AbstractVector{Int64}
    mean_bounds::AbstractMatrix{Float64}
    variance_bounds::AbstractMatrix{Float64}
    max_correlation::Float64
    known_states_bounds::AbstractMatrix{Float64}
end 


"""
state_to_node!(z::AbstractVector{Float64},beleif_state::beleif_state, scaling::scaling)

Over writes the vector z to map from the beleif states space to the scale of the interpolation 
object for the value function. 

z - vector to over write with rescaled values
beleif_state - beleif state oject to rescale
scaling - scaling object that stores data tto define mapping from the beleif state space to the interpolation grid
"""
function state_to_node!(z::AbstractVector{Float64},beleifState,boundary)
    
    mean = beleifState.estimated_states
    covariance = beleifState.covariance_matrix
    known_states = beleifState.known_states
    order = boundary.aproximation_order
    
    # rearange values 
    unobserved_dimensions = length(beleifState.estimated_states)
    known_dimensions = length(beleifState.known_states)
    z[1:unobserved_dimensions] .= (mean.-boundary.mean_bounds[1,:])./(boundary.mean_bounds[2,:].- 
                                  boundary.mean_bounds[1,:]) .* (boundary.aproximation_order[1:unobserved_dimensions].-1).+1
    
    # compute correlation matrix 
    row_number = 0
    total_count =0
    beleifState.correlation_matrix .= cov_to_cor(covariance)
    correlation = beleifState.correlation_matrix
    for n in reverse(1:unobserved_dimensions)
        row_number +=1
        for i in 1:n
            total_count+=1
            index = unobserved_dimensions+total_count
            if row_number > 1 
                
                z[index] = correlation[i+(row_number-1),i]/boundary.max_correlation.* (order[index].-1).+1
                
            else 
                upper_bound = boundary.variance_bounds[2,i]
                lower_bound = boundary.variance_bounds[1,i]
                z[index] = (covariance[i,i].-lower_bound)/ (upper_bound-lower_bound).*(order[index].-1).+1
            end   
        end 
    end
    
    # rescale know state 
    upper = boundary.known_states_bounds[2,:]
    lower = boundary.known_states_bounds[1,:]
    index = unobserved_dimensions+total_count + 1
    z[index:end] .= (known_states.-lower)./(upper.-lower).* (order[index:end].-1).+1
end 

"""
node_to_state(node::AbstractVector{Float64}, scaling::scaling)

Converts a vector node on the scale of the interpolation object to a beleif state 

node - vector on the scale of the interpolation object
scaling - scaling object that stores data tto define mapping from the beleif state space to the interpolation grid
"""
function node_to_state(node::AbstractVector{Float64}, boundary)
    # rearange values 
    unobserved_dimensions = size(boundary.mean_bounds)[2]
    known_dimensions = size(boundary.known_states_bounds)[2]
    order = boundary.aproximation_order
    
    upper = boundary.mean_bounds[2,:]
    lower = boundary.mean_bounds[1,:]
    means = (upper.-lower).*(node[1:unobserved_dimensions].-1)./(order[1:unobserved_dimensions].-1) .+ lower
    
    
    row = 0
    total_count =0
    Cov = zeros(unobserved_dimensions,unobserved_dimensions)
    Cor = zeros(unobserved_dimensions,unobserved_dimensions)
    for n in reverse(1:unobserved_dimensions)
        row +=1
        for i in 1:n
            total_count+=1
            index = total_count + unobserved_dimensions
            if row > 1 # rescale covariance to correlation scale 
                Cov[i+(row-1),i] = sqrt(Cov[i,i]*Cov[i+(row-1),i+(row-1)])*boundary.max_correlation.*(node[index].-1)/(order[index].-1)
                Cov[i,i+(row-1)] = sqrt(Cov[i,i]*Cov[i+(row-1),i+(row-1)])*boundary.max_correlation.*(node[index].-1)/(order[index].-1)
            else # rescale variance }
                upper = boundary.variance_bounds[2,i]
                lower = boundary.variance_bounds[1,i]
                Cov[i,i] = (upper-lower)*(node[index].-1 )/(order[index]-1)+lower
            end   
        end 
    end
    
    index = total_count + unobserved_dimensions + 1
    
    if length(node) < index
        return init_BeleifState(means,Cov,[])
    end 
    
    upper = boundary.known_states_bounds[2,:]
    lower = boundary.known_states_bounds[1,:]
    obs = (upper.-lower).*(node[index:end].-1)./(order[index:end].-1).+lower
    
    return init_BeleifState(means,Cov,obs)
end

"""
value_function

stores data to define value function 

scaling - defines mapping from the beleif state space to the interpolation grid
nodes - beleif state values that correspond to interpolation grid points
values - the value of the value function at the grid points
dims - number of dimensions
interpolation - interpolation object from "Interpolations.jl"
"""
mutable struct ValueFunction
    boundary
    nodes::AbstractVector{}
    values::AbstractVector{Float64}
    dims::Int64
    interpolation
    order
end 


"""
init_value_function(scaling;n_nodes_max = 10^5)

initializes a value function object

m - vector with number of grid point in each dimension of the interpoliation object
mean_bounds - 2 by d matrix, top row has lower bounds bottom row has upper bounds for the mean value of the partially observed variables
variance_bounds - 2 by d matrix, top row has lower bounds bottom row has upper bounds for the variance of the partially observed variables
max_correlation - float giving an upper bound for the covariance 
known_states_bounds - 2 by d matrix, top row has lower bounds bottom row has upper bounds for the value of the observed variables
n_nodes_max- defaul 10^5 keeps function from allocating too much memory unintnetionally 
"""
function init_ValueFunction(aproximation_order,mean_bounds,variance_bounds,max_correlation,known_states_bounds;
                            n_nodes_max = 10^5,order="Cubic")

    boundary = ValueFunctionBoundary(aproximation_order,mean_bounds,variance_bounds,max_correlation,known_states_bounds)
    number_of_nodes = prod(boundary.aproximation_order)
    @assert number_of_nodes < n_nodes_max
    values = zeros(number_of_nodes)
    interpolation = interpolate(reshape(values,ntuple(i->boundary.aproximation_order[i],length(boundary.aproximation_order))), BSpline(Linear()))
#     dimsMean = size(boundary.mean_bounds)[2]
#     dimsobs = size(boundary.known_states_bounds)[2]
    indecies = broadcast(x->[Float64(i) for i in x],reshape(collect(knots(interpolation)),prod(boundary.aproximation_order)))
    nodes = broadcast(i->node_to_state(indecies[i],boundary),1:number_of_nodes)
    
    dims = length(boundary.aproximation_order)
    return ValueFunction(boundary,nodes,values,dims,interpolation,order)
end 


function update!(V)
    dims = Tuple(x for x in V.boundary.aproximation_order)
    if V.order == "Cubic"
        itp = interpolate(reshape(V.values,dims),BSpline(Cubic(Line(OnCell()))))
    elseif V.order == "Quadratic"
        itp = interpolate(reshape(V.values,dims),BSpline(Quadratic(Line(OnCell()))))
    end 
    itp = extrapolate(itp, Line())
        
    V.interpolation = itp
end 

"""
(V::value_function)(nide::AbstractVector{Float64})

evaluates thevalue function at a grid point

node -vector on scalig of the interpolation grid
"""
function (V::ValueFunction)(node::AbstractVector{Float64})
    if V.dims == 2
        return V.interpolation(node[1],node[2])
    elseif V.dims == 3
        return V.interpolation(node[1],node[2],node[3])
    elseif V.dims == 4
        return V.interpolation(node[1],node[2],node[3],node[4])
    elseif V.dims == 5
        return V.interpolation(node[1],node[2],node[3],node[4],node[5])
    elseif V.dims == 6
        return V.interpolation(node[1],node[2],node[3],node[4],node[5],node[6])
    end 
end 


"""
(V::value_function)(z,x::beleif_state)

evaluates the value function at a point in thebeleif state space

z - vector that stoes rescaled beleif state values
x - beleif state value 
"""
function (V::ValueFunction)(z::AbstractVector{Float64},x)#::beleif_state
    
    state_to_node!(z,x, V.boundary)
    
    return V(z)
end 




mutable struct solution
    value_function
    policy_function
end 


mutable struct Model
    total_dimensions_out::Int64 
    total_dimensions_in::Int64 
    total_states::Int64
    num_unobserved_states::Int64 
    num_known_states::Int64 
    num_observations::Int64 
    num_nonadditive_noise::Int64 # number of non-additive noise terms 
    state_transition::Function
    nonadditive_noise_matrix::AbstractMatrix{Float64}
    additive_noise_matrix::Function 
    uses_fixed_policy_function::Bool
    fixed_policy_function::Function
    actions::AbstractVector{}
    reward_function::Function
    discount_factor::Float64
end 


"""
init_model(num_known_states, num_unobserved_states, number_of_observations,
           state_transition, nonadditive_noise_matrix, additive_noise_matrix,
           actions, reward_function, discount_factor)

initializes a model object that defines a MOMDP and sotre all the data
and function required to define the object and compute beleif state tranisiotns

The function has a couple of methods depending on the number of paramters given to it. 
if you do not provide a matrix for the non additive noise terms, of a fixed policy function 
these values will be ignored. 
"""
function init_model(num_known_states,
                    num_unobserved_states,
                    number_of_observations,
                    state_transition,
                    nonadditive_noise_matrix,
                    additive_noise_matrix,
                    fixed_policy_function,
                    actions,
                    reward_function,
                    discount_factor)
    
    @assert (num_known_states+ num_unobserved_states + number_of_observations) == size(additive_noise_matrix(actions[1]))[1]
    
    number_of_process_errors = size(nonadditive_noise_matrix)[1]
    
    Model(num_known_states+ num_unobserved_states + number_of_observations,
          num_unobserved_states + number_of_process_errors,
          num_known_states+ num_unobserved_states,
          num_unobserved_states,
          num_known_states,
          number_of_observations,
          number_of_process_errors,
          state_transition,
          nonadditive_noise_matrix,
          additive_noise_matrix, 
          true, 
          fixed_policy_function,
          actions,
          reward_function,
          discount_factor)
end 


function init_model(num_known_states,
                    num_unobserved_states,
                    number_of_observations,
                    state_transition,
                    nonadditive_noise_matrix,
                    additive_noise_matrix,
                    actions,
                    reward_function,
                    discount_factor)
    
    @assert (num_known_states+ num_unobserved_states + number_of_observations) == size(additive_noise_matrix)[1]
    
    number_of_process_errors = size(nonadditive_noise_matrix)[1]
    
    model(num_known_states+ num_unobserved_states + number_of_observations,
          nnum_unobserved_states + number_of_process_errors,
          num_known_states+ num_unobserved_states,
          num_unobserved_states,
          num_known_states,
          number_of_observations,
          number_of_process_errors,
          state_transition,
          nonadditive_noise_matrix,
          additive_noise_matrix, 
          false, x -> 1,
          reward_function,
          discount_factor)
end 


struct Solver
    estimates_in
    covariance_matrix_in
    estimates_out
    covariance_matrix_out
    state_quadrature
    observation_quadrature
    samples
    covariance_samples
end 

using LinearAlgebra
function init_solver(model,m)
    estimates_in = zeros(model.total_dimensions_in)
    covariance_matrix_in = 1.0*Matrix(I,model.total_dimensions_in,model.total_dimensions_in) 
    estimates_out = zeros(model.total_dimensions_out)
    covariance_matrix_out = 1.0*Matrix(I,model.total_dimensions_out,model.total_dimensions_out) 
    state_quadrature = MvGaussHermite.init_mutable(m,estimates_in,covariance_matrix_in)
    observation_quadrature = MvGaussHermite.init_mutable(m,estimates_out,covariance_matrix_out)
    samples = broadcast(i->zeros(model.total_dimensions_out),1:m^model.total_dimensions_in)
    covariance_samples = broadcast(i->zeros(model.total_dimensions_out,model.total_dimensions_out),1:m^model.total_dimensions_in)
    Solver(estimates_in,covariance_matrix_in,estimates_out,covariance_matrix_out,
           state_quadrature,observation_quadrature,samples,covariance_samples)
end 


"""
time_update(solver,beleifState,action,model)

Updates solver.estimates_out and solver.covariance_matrix_out to acount for the state transitions
and observaiton model. 

solver - Solver object, stores quadrature values 
belierState - BeleifState object
action - choice of action type is dictatbed but model.T
model - Model object, store parameters to define decision problem 
"""
function time_update!(solver,beleifState,action,model)  
    
    # update state estimates 
    solver.covariance_matrix_in[1:model.num_unobserved_states,1:model.num_unobserved_states] .= beleifState.covariance_matrix
    solver.estimates_in[1:model.num_unobserved_states] = beleifState.estimated_states
    
    # add non additive noise covaraince 
    solver.covariance_matrix_in[(model.num_unobserved_states+1):end,(model.num_unobserved_states+1):end] .= model.nonadditive_noise_matrix
    solver.estimates_in[(model.num_unobserved_states+1):end] .= 0.0

    # updated quadrature
    MvGaussHermite.update!(solver.state_quadrature,solver.estimates_in,solver.covariance_matrix_in)

    # propogate samples through state transition 
    if model.uses_fixed_policy_function
        fixed_policy = model.fixed_policy_function(beleifState)
        solver.samples .= broadcast(x->model.state_transition(vcat(beleifState.known_states,x),fixed_policy,action),solver.state_quadrature.nodes)
    else
        solver.samples .= broadcast(x->model.state_transition(vcat(beleifState.known_states,x),fixed_policy,action),solver.state_quadrature.nodes)
    end 

    # compute mean 
    solver.estimates_out .*= 0.0
    for i in 1:length(solver.state_quadrature.nodes)
        solver.estimates_out .+= solver.samples[i]*solver.state_quadrature.weights[i]
    end
    
    # compute covariance 
    solver.covariance_samples .= broadcast(i->(solver.samples[i].-solver.estimates_out).* (solver.samples[i].-solver.estimates_out)',1:length(solver.samples))
    
    solver.covariance_matrix_out .*= 0.0
    for i in 1:length(solver.state_quadrature.nodes)
        solver.covariance_matrix_out .+= solver.covariance_samples[i]*solver.state_quadrature.weights[i]
    end
    solver.covariance_matrix_out .+= model.additive_noise_matrix(action)
    
end 


"""
sampling_distribution!(solver)

updates solver.observation_quadrature with new estimates and covaraince matrix
for obsevered states and observed quantities

solver - Solver object: contains quadrature and intermidiates 
"""
function sampling_distribution!(solver,action,model)
    total_observations = model.num_observations + model.num_known_states
    MvGaussHermite.update!(solver.observation_quadrature,
            solver.estimates_out[1:total_observations],
            solver.covariance_matrix_out[1:total_observations, 1:total_observations)
end 

"""
measurement_update!(observation,action,model)

returns a beleif state object given an observation and the estiamtes states and covariance
matrix stored in the solver object

arguments
beleifState - place holder belefState object
observation - vector of known state and observaiton values
solver - stores estimated values and covaraince matrix 
action - action chosen in a given period
model - stores data 

"""
function measurement_update!(beleifState,observation,solver,action,model)
    # covariance matrix 
    total_observations = model.num_observations + model.num_known_states
    Sigma22 = solver.covariance_matrix_out[(total_observations+1):end,(total_observations+1):end]
    Sigma12 = solver.covariance_matrix_out[1:total_observations,(total_observations+1):end]
    Sigma11 = solver.covariance_matrix_out[1:total_observations,1:total_observations]
    Sigma21 = solver.covariance_matrix_out[(total_observations+1):end,1:total_observations]
    # mean vectors 
    mu1 = solver.estimates_out[1:total_observations]
    mu2 = solver.estimates_out[(total_observations+1):end]
    beleifState.known_states .= observation[1:model.num_known_states]
    beleifState.estimated_states .= mu2 + Sigma21*inv(Sigma11)*(observation .- mu1)
    beleifState.covariance_matrix .= Sigma22 .+ Sigma21 * inv(Sigma11) * Sigma12
end


# next step:
# test measurment update agains KalmanFilters.jl

end # module 