module BeliefStateTransitions

using KalmanFilters
using LinearAlgebra
using Distributions
using TickTock
include("MvGaussHermite.jl")


mutable struct model
    mean::AbstractVector{Float64}
    Cov::AbstractMatrix{Float64}
    actions::AbstractVector{}
    yMean::AbstractVector{Float64}
    yCov::AbstractMatrix{Float64}
    T::Function
    fixed_control::Function
    ct::Float64
    H::AbstractMatrix{Float64}
    SigmaO::Function
    SigmaN::AbstractMatrix{Float64}
    obs_dims::Int64
end 

function init_model(T,fixed_control,H,actions,SigmaO,SigmaN,obs_dims)
    ydims,xdims=size(H)
    mean = zeros(xdims)
    Cov=zeros(xdims,xdims)
    yMean = zeros(ydims)
    yCov=zeros(ydims,ydims)
    return model(mean,Cov,actions,yMean,yCov,T,fixed_control,0.0,H,SigmaO,SigmaN,obs_dims)
end 

function init_model(T,H,actions,SigmaO,SigmaN)
    ydims,xdims=size(H)
    mean = zeros(xdims)
    Cov=zeros(xdims,xdims)
    yMean = zeros(ydims)
    yCov=zeros(ydims,ydims)
    return model(mean,Cov,actions,yMean,yCov,T,x->1,0.0,H,SigmaO,SigmaN,size(H)[1])
end 



mutable struct QuadQuad
    samples::AbstractVector{AbstractVector{Float64}}
    quadrature_x::MvGaussHermite.mutableQuadrature
    weights_x::AbstractVector{Float64}
    quadrature_y::MvGaussHermite.mutableQuadrature
    weights_y::AbstractVector{Float64}
    intermidiate::AbstractVector{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
end


function init_QuadQuad(m_x,m_y,xdims,ydims,obs_dims)
    samples = broadcast(i->zeros(xdims),1:(m_x^xdims))
    quadrature_x = MvGaussHermite.init_mutable(m_x,zeros(xdims),zeros(xdims,xdims))
    weights_x = quadrature_x.weights
    quadrature_y = MvGaussHermite.init_mutable(m_y,zeros(ydims),zeros(ydims,ydims))
    weights_y = quadrature_y.weights
    intermidiate = broadcast(i -> (zeros(obs_dims),zeros(obs_dims,obs_dims)), 1:m_y^ydims)
    return QuadQuad(samples,quadrature_x,weights_x,quadrature_y,weights_y,intermidiate)
end 


function deterministic(quadrature::QuadQuad,state,model)
    # state transitions
    model.mean .= zeros(length(model.mean))
    model.Cov .= model.SigmaN
    model.mean[1:model.obs_dims] .= state[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=state[2]

    # update quadrature
    #print(model.Cov, " ")
    MvGaussHermite.update!(quadrature.quadrature_x,model.mean,model.Cov)
    
    # compute beleif state dependent auxilalry
    model.ct = model.fixed_control(state)

    # propogate samples through state transition 
    quadrature.samples .= broadcast(x->model.T(x,0,model.ct),quadrature.quadrature_x.nodes)
    
    # compute mean 
    model.mean .*= 0
    for i in 1:length(quadrature.quadrature_x.nodes)
        model.mean .+= quadrature.samples[i]*quadrature.weights_x[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (quadrature.samples[i].-model.mean) .*
                            (quadrature.samples[i].-model.mean)', 
                            1:length(quadrature.quadrature_x.nodes))
    
    model.Cov .*= 0
    for i in 1:length(quadrature.quadrature_x.nodes)
        model.Cov .+= samples_Cov[i]*quadrature.weights_x[i]
    end
    model.Cov .+= model.SigmaN
    return model.mean, model.Cov
end 


function integrate!(quadrature::QuadQuad,state,action,model)
    
    # state transitions
    model.mean .= zeros(length(model.mean))
    model.Cov .= model.SigmaN
    model.mean[1:model.obs_dims] .= state[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=state[2]
    
    # update quadrature
    #print(model.Cov, " ")
    MvGaussHermite.update!(quadrature.quadrature_x,model.mean,model.Cov)
    
    # compute beleif state dependent auxilalry
    model.ct = model.fixed_control(state)

    # propogate samples through state transition 
    quadrature.samples .= broadcast(x->model.T(x,action,model.ct),quadrature.quadrature_x.nodes)
    
    # compute mean 
    model.mean .*= 0
    for i in 1:length(quadrature.quadrature_x.nodes)
        model.mean .+= quadrature.samples[i]*quadrature.weights_x[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (quadrature.samples[i].-model.mean) .*
                            (quadrature.samples[i].-model.mean)', 
                            1:length(quadrature.quadrature_x.nodes))
    
    model.Cov .*= 0
    for i in 1:length(quadrature.quadrature_x.nodes)
        model.Cov .+= samples_Cov[i]*quadrature.weights_x[i]
    end
    model.Cov .+= model.SigmaN

    # calcualte distribution of observations 
    model.yMean = model.H * model.mean
    model.yCov = model.H * model.Cov * transpose(model.H) 
    model.yCov .+= model.SigmaO(action,model.ct)
    
    # update quadrature
    MvGaussHermite.update!(quadrature.quadrature_y,model.yMean,model.yCov)
    
    i = 0
    for y in quadrature.quadrature_y.nodes
        i+=1
        mu=KalmanFilters.measurement_update(model.mean,model.Cov,y,model.H,model.SigmaO(action,model.ct))
        quadrature.intermidiate[i]=KalmanFilters.get_state(mu)[1:model.obs_dims],KalmanFilters.get_covariance(mu)[1:model.obs_dims,1:model.obs_dims]
    end 
    
end



function simulate!(x,s,a,model,quad)
    
    # state transitions
    model.mean .= zeros(length(model.mean))
    model.Cov .= model.SigmaN
    model.mean[1:model.obs_dims] .= s[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=s[2]

    # update quadrature
    MvGaussHermite.update!(quad,model.mean,model.Cov)
    
    # compute beleif state dependent auxilalry
    model.ct = model.fixed_control(s)

    
    # propogate samples through state transition 
    samples = broadcast(x->model.T(x,a,model.ct),quad.nodes)
    
    # compute mean 
    model.mean .*= 0
    for i in 1:length(quad.nodes)
        model.mean .+= samples[i]*quad.weights[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-model.mean) .*
                            (samples[i].-model.mean)', 
                            1:length(quad.nodes))
    model.Cov .*= 0
    for i in 1:length(quad.nodes)
        model.Cov .+= samples_Cov[i]*quad.weights[i]
    end
    model.Cov .+= model.SigmaN
    


    x = model.T(x,a, model.ct) .+ rand(Distributions.MvNormal(zeros(length(x)),model.SigmaN))

    y = rand(Distributions.MvNormal(model.H*x, model.SigmaO(a,model.ct)),1)
    
    
    mu=KalmanFilters.measurement_update(model.mean,model.Cov,y,model.H,model.SigmaO(a,model.ct))
    
    s=KalmanFilters.get_state(mu)[1:model.obs_dims],KalmanFilters.get_covariance(mu)[1:model.obs_dims,1:model.obs_dims]
    
    return x,s,y
end


function time_update!(s,model,quad)
    
    # state transitions
    model.mean .= zeros(length(model.mean))
    model.Cov .= model.SigmaN
    model.mean[1:model.obs_dims] .= s[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=s[2]

    # update quadrature
    MvGaussHermite.update!(quad,model.mean,model.Cov)
    
    # compute beleif state dependent auxilalry
    model.ct = model.fixed_control(s)

    
    # propogate samples through state transition 
    samples = broadcast(x->model.T(x,0,model.ct),quad.nodes)
    
    # compute mean 
    model.mean .*= 0
    for i in 1:length(quad.nodes)
        model.mean .+= samples[i]*quad.weights[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-model.mean) .*
                            (samples[i].-model.mean)', 
                            1:length(quad.nodes))
    model.Cov .*= 0
    for i in 1:length(quad.nodes)
        model.Cov .+= samples_Cov[i]*quad.weights[i]
    end
    model.Cov .+= model.SigmaN

    return deepcopy(model.mean),deepcopy(model.Cov)
end


function filter!(y,s,a,model,quad)
    
    # state transitions
    model.mean .= zeros(length(model.mean))
    model.Cov .= model.SigmaN
    model.mean[1:model.obs_dims] .= s[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=s[2]
    
    
    # update quadrature
    MvGaussHermite.update!(quad,model.mean,model.Cov)
    
    # compute beleif state dependent auxilalry
    model.ct = model.fixed_control(s)

 
    # propogate samples through state transition 
    samples = broadcast(x->model.T(x,a,model.ct),quad.nodes)
    
    
    # compute mean 
    model.mean .*= 0
    for i in 1:length(quad.nodes)
        model.mean .+= samples[i]*quad.weights[i]
    end

    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-model.mean) .*
                            (samples[i].-model.mean)', 
                            1:length(quad.nodes))
    model.Cov .*= 0
    for i in 1:length(quad.nodes)
        model.Cov .+= samples_Cov[i]*quad.weights[i]
    end
    
    model.Cov .+= model.SigmaN
    
    
    mu=KalmanFilters.measurement_update(model.mean,model.Cov,y,model.H,model.SigmaO(a,model.ct))
    
    s=KalmanFilters.get_state(mu)[1:model.obs_dims],KalmanFilters.get_covariance(mu)[1:model.obs_dims,1:model.obs_dims]
    
    return s
end




function filter!(s,a,model,quad)
    
    # state transitions
    model.mean .= zeros(length(model.mean))
    model.Cov .= model.SigmaN
    model.mean[1:model.obs_dims] .= s[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=s[2]
    
    
    # update quadrature
    MvGaussHermite.update!(quad,model.mean,model.Cov)
    
    # compute beleif state dependent auxilalry
    model.ct = model.fixed_control(s)

 
    # propogate samples through state transition 
    samples = broadcast(x->model.T(x,a,model.ct),quad.nodes)
    
    
    # compute mean 
    model.mean .*= 0
    for i in 1:length(quad.nodes)
        model.mean .+= samples[i]*quad.weights[i]
    end

    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-model.mean) .*
                            (samples[i].-model.mean)', 
                            1:length(quad.nodes))
    model.Cov .*= 0
    for i in 1:length(quad.nodes)
        model.Cov .+= samples_Cov[i]*quad.weights[i]
    end
    
    model.Cov .+= model.SigmaN
    

    
    s=model.mean,model.Cov
    
    return s
end





## save state transitiosn over value function grid 


mutable struct Transitions2d{T}
    models::AbstractVector{}
    weights::AbstractVector{Float64}
    Filters::AbstractVector{T}
    dimsMean::Int64
    dimsVar::Int64
    dimsAct::Int64
    NMC::Int64
    nodesMean::AbstractVector{Float64}
    nodesVar::AbstractVector{Float64}
    actions::AbstractVector{}
    values::AbstractArray{Float64}
end 


function init_transitions(method,model,V;Nfilter=1000,mQuad=10,NMC=250)
    
    if method == "quadrature"
        Filters = []
        models = []


        for i in 1:Threads.nthreads()

            push!(Filters,init_QuadQuad(mQuad,mQuad,size(model.Cov)[1],size(model.H)[1],model.obs_dims))  
            push!(models,deepcopy(model))

        end 

        meanDims = length(V.nodesMean)
        varDims = length(V.nodesVar)
        actDims = length(model.actions)
        transition = zeros(meanDims,varDims,actDims,mQuad,2)
            
        return Transitions2d{QuadQuad}(models,Filters[1].weights_y,Filters,meanDims,varDims,actDims,mQuad,
                                                V.nodesMean,V.nodesVar,model.actions,transition)
    elseif method == "particle filter"
        Filters = []
        models = []


        for i in 1:Threads.nthreads()
            
            push!(Filters,init_particleFilter(Nfilter,NMC,size(model.Cov)[1],model.obs_dims))  
            push!(models,deepcopy(model))
                    

        end 

        meanDims = length(V.nodesMean)
        varDims = length(V.nodesVar)
        actDims = length(model.actions)
        transition = zeros(meanDims,varDims,actDims,NMC,2)
        weights = ones(NMC)/NMC

        return Transitions2d{particleFilter}(models,weights,Filters,meanDims,varDims,actDims,mQuad,
                                                V.nodesMean,V.nodesVar,model.actions,transition) 
            
    end 
end


function map_state2d!(new_state,state)
    #new_state = zeros(2)
    new_state[1] = state[1][1]
    new_state[2] = state[2][1,1]
    #return new_state
end 

function computeTransitions!(transitions::Transitions2d)
    
    i = transitions.dimsMean
    j = transitions.dimsVar
    k = transitions.dimsAct
    acc = zeros(Threads.nthreads())
    total = i*j*k/Threads.nthreads()
    # 
    Threads.@threads for (indMean,indVar,indAct) in reshape(collect(Iterators.product(1:i,1:j,1:k)),i*j*k)
        
        id = Threads.threadid()
#         if (acc[1] == 0) && (id == 1)
#             tick()
#         end
        
        acc[id] += 1
        
#         if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
#             println("progress = ", round(100*acc[1]/total, digits = 1), "%  ", "time: ", round(peektimer(), digits = 0))
                
#         end 
        
        Cov = [transitions.nodesVar[indVar];;]
        
        BeliefStateTransitions.integrate!(transitions.Filters[id],
                                            ([transitions.nodesMean[indMean]], Cov),
                                            transitions.actions[indAct],
                                            transitions.models[id])
        for n in 1:transitions.NMC
            transitions.values[indMean,indVar,indAct,n,1] = transitions.Filters[id].intermidiate[n][1][1]
            transitions.values[indMean,indVar,indAct,n,2] = transitions.Filters[id].intermidiate[n][2][1,1]
        end 

    end 
end 

        
        

    
mutable struct Transitions{T}
    R::Function
    models::AbstractVector{}
    weights::AbstractVector{Float64}
    Filters::AbstractVector{}
    quads::AbstractVector{}
    stateDims::Int64
    actionDims::Int64
    mQuad::Int64
    nodes::AbstractVector{T}
    actions::AbstractVector{}
    values::AbstractArray{T}
    rewards::AbstractArray{}
end 


function init_Transitions(model,V,R;mQuad=6)
    

    Filters = []
    models = []
    quads = []

    for i in 1:Threads.nthreads()
        push!(quads,MvGaussHermite.init_mutable(mQuad,model.mean,model.Cov)) 
        push!(Filters,init_QuadQuad(mQuad,mQuad,size(model.Cov)[1],size(model.H)[1],model.obs_dims))  
        push!(models,deepcopy(model))

    end 

    stateDims = length(V.nodes)
    actionDims = length(model.actions)
    transitions = Array{typeof(V.nodes[1])}(undef, stateDims, actionDims,length(Filters[1].intermidiate))
    rewards = zeros(stateDims, actionDims)
    
                
    return Transitions{typeof(V.nodes[1])}(R,models,Filters[1].weights_y,Filters,quads,stateDims,actionDims,mQuad,V.nodes,model.actions,transitions,rewards)

end
 

function compute_Transitions!(transitions::Transitions)
    
    i = transitions.stateDims
    j = transitions.actionDims

    acc = zeros(Threads.nthreads())
    total = i*j/Threads.nthreads()
    # 
    Threads.@threads for (i,j) in reshape(collect(Iterators.product(1:i,1:j)),i*j)
        
        id = Threads.threadid()
#         if (acc[1] == 0) && (id == 1)
#             tick()
#         end
        
        acc[id] += 1
        
#         if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
#             println("progress = ", round(100*acc[1]/total, digits = 1), "%  ", "time: ", round(peektimer(), digits = 0))
                
#         end 
        
     
        
        BeliefStateTransitions.integrate!(transitions.Filters[id],transitions.nodes[i],
                                            transitions.actions[j],
                                            transitions.models[id])
        for n in 1:transitions.mQuad
            transitions.values[i,j,n] = deepcopy(transitions.Filters[id].intermidiate[n])
        end 

    end 
end 
    
function computeRewards!(transitions::Transitions)
    
    i = transitions.stateDims
    j = transitions.actionDims
    acc = zeros(Threads.nthreads())
    total = i*j/Threads.nthreads()
    # 
    Threads.@threads for (stateInd,actionInd) in reshape(collect(Iterators.product(1:i,1:j)),i*j)
            
        id = Threads.threadid()
#         if (acc[1] == 0) && (id == 1)
#             tick()
#         end
                
        acc[id] += 1
        
#         if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
#             println("progress = ", round(100*acc[1]/total, digits = 1), "%  ", "time: ", round(peektimer(), digits = 1))
                
#         end 
            
        transitions.models[id].ct = transitions.models[id].fixed_control(transitions.nodes[stateInd])
            
        MvGaussHermite.update!(transitions.quads[id],transitions.nodes[stateInd][1],transitions.nodes[stateInd][2])  
            
        transitions.rewards[stateInd,actionInd] = sum(broadcast(x -> transitions.R(x,transitions.actions[actionInd],transitions.models[id].ct), 
                                                                            transitions.quads[id].nodes).* transitions.quads[id].weights)
            
    end 
end  
    
    
        
mutable struct Rewards2d
    R::Function
    quads
    models
    dimsMean::Int64
    dimsVar::Int64
    dimsAct::Int64
    nodesMean::AbstractVector{Float64}
    nodesVar::AbstractVector{Float64}
    actions::AbstractVector{}
    values::AbstractArray{Float64}
end 
    



function init_rewards(R,model,V;mQuad=25)
    
    quads = []
    models = []
    for i in 1:Threads.nthreads()
        push!(quads,MvGaussHermite.init_mutable(mQuad,model.mean,model.Cov)) 
        push!(models,deepcopy(model))  
    end 
    meanDims = length(V.nodesMean)
    varDims = length(V.nodesVar)
    actDims = length(model.actions)
    values = zeros(meanDims,varDims,actDims)

    return Rewards2d(R,quads,models,meanDims,varDims,actDims,V.nodesMean,V.nodesVar,model.actions,values)
end
       
    
function simulateRewards!(x,s,a,Rewards)
            
    Rewards.models[1].ct = Rewards.models[1].fixed_control(s)

    return Rewards.R(x,a,Rewards.models[1].ct)
                                                                
end 
        
function computeRewards!(Rewards::Rewards2d)
    
    i = Rewards.dimsMean
    j = Rewards.dimsVar
    k = Rewards.dimsAct
    acc = zeros(Threads.nthreads())
    total = i*j*k/Threads.nthreads()
    # 
    Threads.@threads for (indMean,indVar,indAct) in reshape(collect(Iterators.product(1:i,1:j,1:k)),i*j*k)
            
        id = Threads.threadid()
            
            
        acc[id] += 1
            
        
        Rewards.models[id].ct = Rewards.models[id].fixed_control(([Rewards.nodesMean[indMean]],[Rewards.nodesVar[indVar];;]))
            
        MvGaussHermite.update!(Rewards.quads[id],[Rewards.nodesMean[indMean]],[Rewards.nodesVar[indVar];;])  
            
        Rewards.values[indMean,indVar,indAct] = sum(broadcast(x -> Rewards.R(x,Rewards.actions[indAct],Rewards.models[id].ct), 
                                                                            Rewards.quads[id].nodes).* Rewards.quads[id].weights)
            
    end 
end   
      
        
        

    
 
        
##3 bonus stuff
        
#         """
#     integrate!(quadrature::MvGaussHermite.quadrature,state,action,model)

# quadrature - object that defines the gausian quadrature nodes and stores 
# """
# function integrate!(quadrature::gausianQuad,state,action,model)
    
#     # state transitions
#     model.mean = zeros(length(model.mean))
#     model.Cov = model.SigmaN
#     model.mean[1:model.obs_dims] .= state[1]
#     model.Cov[1:model.obs_dims,1:model.obs_dims].=state[2]
#     tu = KalmanFilters.time_update(model.mean,model.Cov,x->model.T(x,action[1]),model.SigmaN)
#     model.mean,model.Cov=KalmanFilters.get_state(tu), KalmanFilters.get_covariance(tu)
    
#     # calcualte distribution of observations 
#     model.yMean = model.H * model.mean
#     model.yCov = model.H * model.Cov * transpose(model.H) 
#     model.yCov .+= model.SigmaO(action[1],action[2])
    
#     # update quadrature
#     MvGaussHermite.update!(quadrature.quadrature,model.yMean,model.yCov)
    
#     i = 0
#     for y in quadrature.quadrature.nodes
#         i+=1
#         mu=KalmanFilters.measurement_update(model.mean,model.Cov,y,model.H,model.SigmaO(action[1],action[2]))
#         quadrature.intermidiate[i]=KalmanFilters.get_state(mu)[1:model.obs_dims], KalmanFilters.get_covariance(mu)[1:model.obs_dims,1:model.obs_dims]
#     end 
    
# end 



# function integrate!(quadrature::particleQuad,state,action,model)
    
#     # state transitions
#     model.mean .= zeros(length(model.mean))
#     model.Cov .= model.SigmaN
#     model.mean[1:model.obs_dims] .= state[1]
#     model.Cov[1:model.obs_dims,1:model.obs_dims].=state[2]

    
#     d=Distributions.MvNormal(model.mean,model.Cov)
#     quadrature.samples .= rand(d,quadrature.Nfilter)
#     dN = Distributions.MvNormal(zeros(length(model.mean)),model.SigmaN)
    
#     # propogate samples through state transition 
#     quadrature.samples .= mapslices(x->model.T(x,action[1]).+rand(dN),quadrature.samples,dims =1)
    
#     # compute mean 
#     model.mean .*= 0
#     for i in 1:quadrature.Nfilter
#         model.mean .+= quadrature.samples[:,i]/quadrature.Nfilter
#     end
    
#     # compute covariance 
#     samples_Cov = broadcast(i -> (quadrature.samples[:,i].-model.mean) .*
#                             (quadrature.samples[:,i].-model.mean)', 
#                             1:quadrature.Nfilter)
#     model.Cov .*= 0
#     for i in 1:quadrature.Nfilter
#         model.Cov .+= samples_Cov[i]/quadrature.Nfilter
#     end
    
    
#     # calcualte distribution of observations 
#     model.yMean = model.H * model.mean
#     model.yCov = model.H * model.Cov * transpose(model.H) 
#     model.yCov .+= model.SigmaO(action[1],action[2])
    
#     # update quadrature
#     MvGaussHermite.update!(quadrature.quadrature,model.yMean,model.yCov)
    
#     i = 0
#     for y in quadrature.quadrature.nodes
#         i+=1
#         mu=KalmanFilters.measurement_update(model.mean,model.Cov,y,model.H,model.SigmaO(action[1],action[2]))
#         quadrature.intermidiate[i]=KalmanFilters.get_state(mu)[1:model.obs_dims],KalmanFilters.get_covariance(mu)[1:model.obs_dims,1:model.obs_dims]
#     end 
    
# end
        
        
        
# mutable struct gausianQuad
#     quadrature::MvGaussHermite.mutableQuadrature
#     weights::AbstractVector{Float64}
#     intermidiate::AbstractVector{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
# end 

# function init_quadrature(m,xdims,ydims,obs_dims)
#     #mu,Cov=zeros(xdims),zeros(xdims,xdims)
#     quadrature = MvGaussHermite.init_mutable(m,zeros(ydims),zeros(ydims,ydims))
#     intermidiate = broadcast(i -> (zeros(obs_dims),zeros(obs_dims,obs_dims)), 1:m^ydims)
#     return gausianQuad(quadrature,quadrature.weights,intermidiate)
# end 

mutable struct particleFilter
    Nfilter::Int64
    Nintegrate::Int64
    weightsFilter::AbstractVector{Float64}
    weights::AbstractVector{Float64}
    samples::AbstractMatrix{Float64}
    intermidiate::AbstractVector{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
end

function init_particleFilter(Nfilter,Nintegrate,xdims,obs_dims)
    @assert Nintegrate <= Nfilter
    samples = zeros(xdims,Nfilter)
    weightsFilter = zeros(Nfilter)
    weights = 1.0*ones(Nintegrate)
    intermidiate = broadcast(i -> (zeros(obs_dims),zeros(obs_dims,obs_dims)), 1:Nintegrate)
    return particleFilter(Nfilter,Nintegrate,weightsFilter,weights,samples,intermidiate)
end 

function integrate!(quadrature::particleFilter,state,action,model)
    

    # state transitions
    model.mean = zeros(length(model.mean))
    model.Cov = model.SigmaN
    model.mean[1:model.obs_dims] .= state[1]
    model.Cov[1:model.obs_dims,1:model.obs_dims].=state[2]
    
    d=Distributions.MvNormal(model.mean,model.Cov)
    quadrature.samples .= rand(d,quadrature.Nfilter)
    dN = Distributions.MvNormal(zeros(length(model.mean)),model.SigmaN)
    
    # auxillary
    model.ct = model.fixed_control(state)
        
    # propogate samples through state transition 
    quadrature.samples .= mapslices(x->model.T(x,action,model.ct).+rand(dN),quadrature.samples,dims =1)
    
   
    # compute observations
    obserror = Distributions.MvNormal(zeros(size(model.H)[1]),model.SigmaO(action,model.ct))
    ynoise = rand(obserror,quadrature.Nintegrate)
    
    # compute observation 
    y = broadcast(i -> model.H * quadrature.samples[:,i] .+ ynoise[:,i], 1:quadrature.Nintegrate)
    
    # compute updated states 
    j = 0
    for yi in y
        j+=1
        # compute weights
        quadrature.weightsFilter .= broadcast(i -> pdf(obserror,yi - model.H *quadrature.samples[:,i]), 1:quadrature.Nfilter)
        quadrature.weightsFilter .*= 1/sum(quadrature.weightsFilter)
        # compute mean 
        quadrature.intermidiate[j][1] .*= 0.0
        for i in 1:quadrature.Nfilter
            quadrature.intermidiate[j][1] .+= quadrature.weightsFilter[i]*quadrature.samples[1:model.obs_dims,i]
        end

        # compute covariance 
        samples_Cov = broadcast(i -> (quadrature.samples[:,i].-quadrature.intermidiate[j][1]) .*
                                (quadrature.samples[:,i].-quadrature.intermidiate[j][1])', 
                                1:quadrature.Nfilter)
        quadrature.intermidiate[j][2] .*= 0.0
        for i in 1:quadrature.Nfilter
            quadrature.intermidiate[j][2] .+= (quadrature.weightsFilter[i]*samples_Cov[i])[1:model.obs_dims,1:model.obs_dims]
        end

    end  
end 
# mutable struct particleQuad
#     Nfilter::Int64
#     samples::AbstractMatrix{Float64}
#     quadrature::MvGaussHermite.mutableQuadrature
#     weights::AbstractVector{Float64}
#     intermidiate::AbstractVector{Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}
# end


# function init_particleQuad(Nfilter,m,xdims,ydims,obs_dims)
#     samples = zeros(xdims,Nfilter)
#     quadrature = MvGaussHermite.init_mutable(m,zeros(ydims),zeros(ydims,ydims))
#     weights = quadrature.weights
#     intermidiate = broadcast(i -> (zeros(obs_dims),zeros(obs_dims,obs_dims)), 1:m^ydims)
#     return particleQuad(Nfilter,samples,quadrature,weights,intermidiate)
# end 



    
# tools for analysis   
include("ParticleFilters.jl")
function simulation(x0,s0,T,filter,P,model,R)
    xls = []
    sls = []
    als = []
    cls = []
    rls = []
    yls=[]
    #dsnls = []
        
    st = s0
    xt = x0
    yt = x0
    for t in 1:T
        # compute belief state and actions 
        st = ParticleFilters.mean_cov(filter)
        ct = model.fixed_control(st)
        at = P([st[1][1],st[2][1,1]]) # does not generalize past 1d POMDP
            
        # update lists with current states and actions 
        push!(yls,yt)
        push!(xls,xt)
        push!(sls,st)
        push!(als,at)
        push!(rls,R(xt,at,ct))
        push!(cls,ct)
        #push!(dsnls,deepcopy(filter.samples))
            
        # time update 
        xt = model.T(xt,at,ct) .+ rand(Distributions.MvNormal(zeros(size(model.SigmaN)[1]),model.SigmaN))
        ParticleFilters.time_update!(filter,model,at,ct)
        
        # measurement update
        yt = rand(Distributions.MvNormal(model.H*xt, model.SigmaO(at,ct)))
        status = ParticleFilters.measurement_update!(filter,yt,model,at,ct)
        if status == "success"           
            ParticleFilters.resample!(filter)
        else
            print(status)
            return "failed"
        end 
        
        push!(yls,yt)
        
    end
    return xls,sls,als,cls,rls,yls#,dsnls
end 

    
function simulation_kf(x0,s0,T,P,model,R,quad)
    xls = []
    sls = []
    als = []
    cls = []
    rls = []
    yls=[]
    #dsnls = []
        
    st = s0
    xt = x0
    yt = x0
    for t in 1:T
        # compute belief state and actions 
        ct = model.fixed_control(st)
        at = P([st[1][1],st[2][1,1]]) # does not generalize past 1d POMDP
        
        # update lists with current states and actions 
        push!(yls,yt)
        push!(xls,xt)
        push!(sls,st)
        push!(als,at)
        push!(rls,R(xt,at,ct))
        push!(cls,ct)
        #push!(dsnls,deepcopy(filter.samples))
        
        xt,st,yt=simulate!(xt,st,at,model,quad)
        
    end
    return xls,sls,als,cls,rls,yls
end 
        
        

    
end 