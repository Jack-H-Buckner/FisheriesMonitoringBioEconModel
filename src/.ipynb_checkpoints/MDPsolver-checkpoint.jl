module MDPsolver

include("MvGaussHermite.jl")
include("ValueFunctions.jl")
include("BeliefStateTransitions.jl")
struct objective
    delta::Float64
    R::Function 
end 


function value_expectation!(z,actInd::Int,meanInd::Int,varInd::Int,grid,V)
    sum(mapslices( x->V(z,x), grid.values[meanInd,varInd,actInd,:,:], dims = 2).*grid.weights)
end


function bellman!(z,meanInd::Int,varInd::Int,grid,V,rewards,delta)
    vals =broadcast(i->rewards.values[meanInd,varInd,i]+delta*value_expectation!(z,i,meanInd,varInd,grid,V), 1:grid.dimsAct)
    return vals[argmax(vals)]
end 

function policy!(z,meanInd::Int,varInd::Int,grid,V,rewards,delta)
    vals =broadcast(i->rewards.values[meanInd,varInd,i]+delta*value_expectation!(z,i,meanInd,varInd,grid,V), 1:grid.dimsAct)
    return grid.actions[argmax(vals)]
end

function bellman!(z,actions,meanInd::Int,varInd::Int,grid,V,rewards,delta)
    vals =broadcast(i->rewards.values[meanInd,varInd,i]+delta*value_expectation!(z,i,meanInd,varInd,grid,V), actions)
    return vals[argmax(vals)]
end 

function policy!(z,actions, meanInd::Int,varInd::Int,grid,V,rewards,delta)
    vals =broadcast(i->rewards.values[meanInd,varInd,i]+delta*value_expectation!(z,i,meanInd,varInd,grid,V), actions)
    return grid.actions[argmax(vals)]
end

function solve_parallel(grid,V,rewards, delta;threashold=10^-5,max_iter=500,verbos = true)
    test = 10*length(V.values)
    z = broadcast(i->zeros(2), 1:Threads.nthreads())
    grids = broadcast(i->deepcopy(grid), 1:Threads.nthreads())
    rewards = broadcast(i->deepcopy(rewards), 1:Threads.nthreads())
    n = 0   
    
    while (test > length(V.values)*threashold) & (n < max_iter)
        n+=1
        if verbos
            print("interation: ", n)
            println("  convergence: ", test)
        end 
        
        acc = 0
        Threads.@threads for inds in reshape(collect(Iterators.product(1:grid.dimsMean,1:grid.dimsVar)),
                                                grid.dimsVar*grid.dimsMean)
            i,j=inds
            k = (j-1)*grid.dimsVar + i

            id = Threads.threadid()
            B = bellman!(z[id],i,j,grids[id],V,rewards[id], delta) #bellman!(zeros(2),i,j,grid,V,objective,xQuad)
            acc += (V.values[k] - B)^2
            V.values[k] = B
        end

        ValueFunctions.update1!(V)
        test = acc
    end 
end 


function policy_parallel!(P::ValueFunctions.Norm2DGrid_policy,grid,V,rewards, delta;threashold = 10^-2)
 
    z = broadcast(i->zeros(2), 1:Threads.nthreads())
    #xQuad = broadcast(i->deepcopy(xQuad), 1:Threads.nthreads())   

    Threads.@threads  for inds in reshape(collect(Iterators.product(1:grid.dimsMean,1:grid.dimsVar)),grid.dimsVar*grid.dimsMean)
        i,j=inds
        k = (j-1)*grid.dimsVar + i
        id = Threads.threadid()
        policy = policy!(z[id],i,j,grid,V,rewards, delta)

        P.action_values[k] = policy[1]
        P.observation_values[k] = policy[2]
    end

    ValueFunctions.update!(P)

end

function policy_parallel!(P::ValueFunctions.Norm2DGrid_obs_policy,grid,V,rewards, delta;threashold = 10^-2)
  
    z = broadcast(i->zeros(2), 1:Threads.nthreads())
    #xQuad = broadcast(i->deepcopy(xQuad), 1:Threads.nthreads())   

    Threads.@threads  for inds in reshape(collect(Iterators.product(1:grid.dimsMean,1:grid.dimsVar)),grid.dimsVar*grid.dimsMean)
        i,j=inds
        k = (j-1)*grid.dimsVar + i
        id = Threads.threadid()
        policy = policy!(z[id],i,j,grid,V,rewards, delta)
        #P.action_values[k] = policy[1]
        P.observation_values[k] = policy
    end

    ValueFunctions.update!(P)

end

function simulation(x0,s0,T,P,model,Rewards;m = 10)
    quad = MvGaussHermite.init_mutable(m,zeros(size(model.SigmaN)[1]),model.SigmaN)
    xls = []
    sls = []
    als = []
    auxls = []
    rls = []
    yls=[]
    s = s0
    x = x0
    for t in 1:T
        a = P([s[1][1],s[2][1,1]])
        push!(xls,x)
        push!(sls,s)
        push!(als,a)
        push!(rls, BeliefStateTransitions.simulateRewards!(x,s,a,Rewards))
        push!(auxls,model.fixed_control(s))
        x,s,y=BeliefStateTransitions.simulate!(x,s,a,model,quad)
        push!(yls,y)
    end
    return xls,sls,als,auxls,rls,yls
end 





#### alternative methods


function value_expectation!(z,actionInd::Int,stateInd::Int,grid,V)
    sum(broadcast( x->V(z,x), grid.values[stateInd,actionInd,:]).*grid.weights)
end


function bellman!(z,stateInd::Int,grid,V,delta)
    vals =broadcast(i->grid.rewards[stateInd,i]+delta*value_expectation!(z,i,stateInd,grid,V), 1:grid.actionDims)
    return vals[argmax(vals)]
end 

function policy!(z,stateInd::Int,grid,V,delta)
    vals =broadcast(i->grid.rewards[stateInd,i]+delta*value_expectation!(z,i,stateInd,grid,V), 1:grid.actionDims)
    return grid.actions[argmax(vals)]
end


function solve_parallel(grid,V,delta;threashold=10^-3,max_iter=200,verbos = true)
    test = 10*length(V.nodes)
    z = broadcast(i->zeros(5), 1:Threads.nthreads())
    #grids = broadcast(i->deepcopy(grid), 1:Threads.nthreads())
    #rewards = broadcast(i->deepcopy(rewards), 1:Threads.nthreads())
    n = 0   
    
    while (test > threashold*length(V.nodes)) & (n < max_iter)
        n+=1
        if verbos
            print("interation: ", n)
            println("  convergence: ", test)
        end 
        
        acc = 0
        Threads.@threads for stateInd in 1:length(V.nodes)

            id = Threads.threadid()
            B = bellman!(z[id],stateInd,grid,V, delta) #bellman!(zeros(2),i,j,grid,V,objective,xQuad)
            acc += (V.values[stateInd] - B)^2
            V.values[stateInd] = B
        end

        ValueFunctions.update2!(V)
        test = acc
    end 
end 



# continuous action variable
using Optim


struct discrete_continuous_actions{T}
    discrete_levels::AbstractVector{}
    continuous_upper_bound::T
    continuous_lower_bound::T
    continuous_initial_guess::T
end


mutable struct discrete_continuous_solver
    z
    belief_state_filter
    rewards_quadrature
    model
    delta
    actions
    discrete_action_levels
    continuous_action_upper_bound
    continuous_action_lower_bound
    continuous_action_initial_guess
end 

function value_expectation!(discrete_continuous_solver,state,action,V)
    
    BeliefStateTransitions.integrate!(discrete_continuous_solver.belief_state_filter,state,action,discrete_continuous_solver.model)
    
    Vprime = sum(broadcast( x->V(discrete_continuous_solver.z,x), discrete_continuous_solver.belief_state_filter.intermidiate)
                            .*discrete_continuous_solver.belief_state_filter.weights_y)
    
    R = sum(broadcast(x -> discrete_continuous_solver.model.R(x,action,models.ct),discrete_continuous_solver.rewards_quadrature.nodes)
                            .* discrete_continuous_solver.rewards_quadrature.weights)
    
    return R + discrete_continuous_solver.delta*Vprime
end


function bellman!(discrete_continuous_solver,state,V)
    
    vals = zeros(length(discrete_continuous_solver.discrete_levels))
        
    i=0  
    
    for discrete_action in discrete_continuous_solver.discrete_levels
        i+=1
        f = continuous_action -> -1*value_expectation!(discrete_continuous_solver,state,action(continuous_action,discrete_action),V)
        
        # call bounded interval search if apropreate 
        if typeof(ddiscrete_continuous_solver.continuous_action_upper_bound) == Float64
            sol = Optim.optimize(f,discrete_continuous_solver.continuous_action_lower_bound, 
                                 discrete_continuous_solver.continuous_action_upper_bound)
            
        # call bounded region search
        else
            sol = Optim.optimize(f,discrete_continuous_solver.continuous_action_lower_bound, 
                                 discrete_continuous_solver.continuous_action_upper_bound,
                                 discrete_continuous_solver.continuous_action_initial_guess)
        end 
        
        vals[i] = -1*sol.minimum
    end 
    
    return vals[argmax(vals)]
end 



function policy!(discrete_continuous_solver,state,V)
    
    vals = zeros(length(discrete_continuous_solver.discrete_levels))
    
    optimum_continuous_actions = []
        
    i=0  
    
    for discrete_action in discrete_continuous_solver.discrete_levels
        i+=1
        f = continuous_action -> -1*value_expectation!(discrete_continuous_solver,state,action(continuous_action,discrete_action),V)
        
        # call bounded interval search if apropreate 
        if typeof(ddiscrete_continuous_solver.continuous_action_upper_bound) == Float64
            sol = Optim.optimize(f,discrete_continuous_solver.continuous_action_lower_bound, 
                                 discrete_continuous_solver.continuous_action_upper_bound)
            
        # call bounded region search
        else
            sol = Optim.optimize(f,discrete_continuous_solver.continuous_action_lower_bound, 
                                 discrete_continuous_solver.continuous_action_upper_bound,
                                 discrete_continuous_solver.continuous_action_initial_guess)
        end 
        
        vals[i] = -1*sol.minimum
        
        push!(optimum_continuous_actions,sol.minimizer)
    end 
    
    action = (optimum_continuous_actions[argmax(vals)], actions.discrete_levels[argmax(vals)])
    return action
end 



function solve_parallel(discrete_continuous_solver,V;threashold=10^-5,max_iter=200,verbos = true)
    test = 10*length(V.nodes)
    z = broadcast(i->zeros(5), 1:Threads.nthreads())
        
    solvers = broadcast(i->deepcopy(discrete_continuous_solver), 1:Threads.nthreads())
 
    n = 0   
    
    while (test > threashold*length(V.nodes)) & (n < max_iter)
        n+=1
        if verbos
            print("interation: ", n)
            println("  convergence: ", test)
        end 
        
        acc = 0
        Threads.@threads for i in 1:length(V.nodes)

            state = V.nodes[i]
            id = Threads.threadid()
            B = bellman!(solvers[id],state,V) 
            acc += (V.values[stateInd] - B)^2
            V.values[stateInd] = B
        end

        ValueFunctions.update2!(V)
        test = acc
    end 
end 


end