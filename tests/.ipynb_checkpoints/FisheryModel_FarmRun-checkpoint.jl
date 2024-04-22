include("../examples/BiomassDynamics2.jl")
include("../src/BeleifMDPSolvers.jl")
using JLD2


# set parameter values 
pars = BiomassDynamics2.pars_short

# define state transition, reward and observaiton function 
T!(x,Ht) = BiomassDynamics2.deterministic!(x,Ht,pars)[1]
T(x,Ht) = BiomassDynamics2.deterministic(x,Ht,pars)[1]
function R_obs(x,Ht)
    Fmax = 0.5
    B = exp(x[1])
    if B > Ht  
        s = [Fmax,-log((B-Ht)/B)]
        F = s[argmin(s)]
    else
        F = Fmax
    end
    H = (1-exp(-F))*B
    return H 
end

R(x,Ht,Ot) = R_obs(x,Ht) - 8.0*Ot
H = [0.0 1.0 0.0 ; -1.0 1.0 0.0]
sigma_R = 0.5
Sigma_N = [10^-6 0 0; 0 10^-6 0; 0 0 sigma_R]

function Sigma_O(a, obs)
    if obs == 2
        return [0.1 0 ; 0 0.025]
    elseif obs == 1
        return [0.5 0 ; 0 0.1]
    else
        return [4.0 0 ; 0 0.5]
    end
end 

# dfine discount rate, action space and grid range 
delta= 1/(1+0.05)
actions = collect(0.0:0.25:10.0).^2
observations = [0,1,2]
lower_mu = log.([5.0,0.25])
upper_mu = log.([250,0.5])


# initialize solver 
solver=BeleifMDPSolvers.initSolver(T!,T,R,R_obs,
            H,Sigma_N,Sigma_O,delta,
            actions, observations,
            lower_mu,upper_mu;n_grids_obs = 20,n_grid = 6)
# print number of nodes used
length(solver.bellmanIntermidiate)


# solve observed system 
BeleifMDPSolvers.solve_observed_parallel!(solver;tol = 10^-4)
# run full solution 
BeleifMDPSolvers.solve_parallel!(solver;max_iter = 1, tol = 10^-4)

@save "data/FisheryModel_solution_1.jld2"  solver