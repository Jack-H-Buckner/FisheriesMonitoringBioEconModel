module RelativeAbundance


include("../src/MvGaussHermite.jl")
include("../src/utils.jl")
#include("../src/ValueFunctions.jl")
using Distributions
using Interpolations

using TickTock

### state transitions ###

function convert_harvest(B,H,Fmax)

    Smin = -log(1 - Fmax)

    
    if B > H
        s = [-log((B-H)/B), Smin]
        S = s[argmin(s)]
    else
        S = Smin
    end 
    
    H = B*(1-exp(-S))
    return H, B*exp(-S)
    
end 

pars = (1.427,-0.2383, 0.8, [0.05 0.0; 0.0 0.0000001], 5, 0.0085, 0.0,25.0)#10, 0.0085, 0.0,25.0
function Bt(xobs,xpartial,Ht,pars)

    B = exp(xobs[1]+xpartial[1])
    Binfty = exp(xpartial[1])
    r= exp(pars[1]+pars[2]*xpartial[1]) 

    H,B=convert_harvest(B, Ht, pars[3])

    Bprime = r*B- (r-1)*B^2/Binfty

    if Bprime <0
        Bprime = 1.0
    end 

    return [log(Bprime)-xpartial[1], xpartial[1]]
    
end 

mutable struct intermidiate
    quadx
    quady
    mean::AbstractVector{Float64}
    Cov::AbstractMatrix{Float64}
    ymean::AbstractVector{Float64}
    yCov::AbstractMatrix{Float64}
    states::AbstractVector{}
end 

function init(m)
    quady = MvGaussHermite.init_mutable(m, [0.0],[1.0;;])
    quadx = MvGaussHermite.init_mutable(m, [0.0],[1.0;;])
    s = ([0.0], [0.0],[1.0;;])
    states = Array{typeof(s)}(undef,m)
    return intermidiate(quadx,quady,[0.0, 0.0],[1.0 0.0; 0.0 1.0],[0.0],[1.0;;],states)
end 

function state_transition!(intermidiate,s,Ht,pars)
    xobs = s[1]
    MvGaussHermite.update!(intermidiate.quadx,s[2],s[3])
    
    # updated states
    samples = broadcast(x->Bt(xobs,x,Ht,pars),intermidiate.quadx.nodes)

    # compute mean 
    intermidiate.mean .*= 0
    for i in 1:length(intermidiate.quadx.nodes)
        intermidiate.mean .+= samples[i]*intermidiate.quadx.weights[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-intermidiate.mean) .*
                            (samples[i].-intermidiate.mean)', 
                            1:length(intermidiate.quadx.nodes))
    intermidiate.Cov .*= 0
    for i in 1:length(intermidiate.quadx.nodes)
        intermidiate.Cov .+= samples_Cov[i]*intermidiate.quadx.weights[i]
    end

    intermidiate.Cov .+= pars[4]

    
    intermidiate.ymean = [1.0 0.0] * intermidiate.mean
    intermidiate.yCov = [1.0 0.0] * intermidiate.Cov * transpose([1.0 0.0])
    
    # update quadrature
    MvGaussHermite.update!(intermidiate.quady,intermidiate.ymean,intermidiate.yCov)
    rho = intermidiate.Cov[1,2] /sqrt(intermidiate.Cov[1,1]*intermidiate.Cov[2,2])
    i = 0
    for y in intermidiate.quady.nodes
        i+=1
        mu = intermidiate.mean[2] + rho*sqrt(intermidiate.Cov[2,2])*(y[1] -intermidiate.mean[1])/sqrt(intermidiate.Cov[1,1])
        sigma = (1-rho^2)*intermidiate.Cov[2,2]
        intermidiate.states[i] = (y,[mu],[sigma;;])
    end 
    
end 


function state_transition_sim!(intermidiate,x,s,Ht,pars)
    
    xt = Bt([x[1]],[x[2]],Ht,pars) .+ rand(Distributions.MvNormal([0.0, 0.0], pars[4]))
    
    xobs = s[1]
    MvGaussHermite.update!(intermidiate.quadx,s[2],s[3])
    
    # updated states
    samples = broadcast(x->Bt(xobs,x,Ht,pars),intermidiate.quadx.nodes)

    # compute mean 
    intermidiate.mean .*= 0
    for i in 1:length(intermidiate.quadx.nodes)
        intermidiate.mean .+= samples[i]*intermidiate.quadx.weights[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-intermidiate.mean) .*
                            (samples[i].-intermidiate.mean)', 
                            1:length(intermidiate.quadx.nodes))
    intermidiate.Cov .*= 0
    for i in 1:length(intermidiate.quadx.nodes)
        intermidiate.Cov .+= samples_Cov[i]*intermidiate.quadx.weights[i]
    end
    intermidiate.Cov .+= pars[4]
    
    intermidiate.ymean = [1.0 0.0] * intermidiate.mean
    intermidiate.yCov = [1.0 0.0] * intermidiate.Cov * transpose([1.0 0.0])
    rho = intermidiate.Cov[1,2] /sqrt(intermidiate.Cov[1,1]*intermidiate.Cov[2,2])
    
 
    mean = intermidiate.mean[2] + rho*sqrt(intermidiate.Cov[2,2])*(xt[1] -intermidiate.mean[1])/sqrt(intermidiate.Cov[1,1])
    Var = (1-rho^2)*intermidiate.Cov[2,2]

    return xt,([xt[1]],[mean],[Var;;])
    
end 

### rewards ###

function R(x,actions,pars)
    Ht = actions
    B = exp(x[1]+x[2])
    H,B=convert_harvest(B, Ht, pars[3])
    return H-pars[5]*H/B-pars[6]*H^2 + pars[7]*B/(1+pars[7]/pars[8]*B)
end


### value function ###

mutable struct V
    z::AbstractVector{Float64}
    m::Int64
    upper_x::Float64
    lower_x::Float64
    upper_mu::Float64
    lower_mu::Float64
    upper_sigma::Float64
    grid::AbstractArray{AbstractVector{Float64}}
#     xNodes::AbstractVector{Float64}
#     meanNodes::AbstractVector{Float64}
#     sigmaNodes::AbstractVector{Float64}
    nodes::AbstractVector{}
    values::AbstractVector{Float64}
    interpolation
end 
    
    
function init_V(upper_x,lower_x,upper_mu,lower_mu,upper_sigma,m)

    d = 3
    # calcualte nodes
    z = collect((0.5/m):1/m:(1-0.5/m))
    a = [lower_x,lower_mu, 0.0]
    b = [upper_x,upper_mu, upper_sigma]
    nodes = z .*transpose(b.-a) #.- a
    
    nodes = mapslices(x-> x .+ a, nodes, dims = 2)

    z = (z.+1)./2 .*transpose(repeat([2.0],d))
    z = mapslices(x-> x .-  1.0, z, dims = 2)
    grid = utils.collect_nodes(nodes) # nodes on desiered domain
    # initialize values as zero
    values = 1.0*zeros(m^d) #
    values_ = reshape(values,ntuple(_->m,d))
    itp = Interpolations.interpolate(values_,Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnCell()))))
    itp = Interpolations.extrapolate(itp, Interpolations.Line())
    
    Nodes = Array{Tuple{AbstractVector{Float64},AbstractVector{Float64},AbstractMatrix{Float64}}}(undef, length(values))
    i = 0
    for sigma in nodes[:,3]
        for mu in  nodes[:,2]
            for x in nodes[:,1]
                i+=1
                Nodes[i] = ([x], [mu], [sigma;;])
            end
        end
    end
    
    return V(zeros(3),m,upper_x,lower_x,upper_mu,lower_mu,upper_sigma,grid,Nodes,values,itp)
    
end 
    
    
function (p::V)(s) 
        

    p.z[1] = ((s[1][1]-p.lower_x)/(p.upper_x-p.lower_x))*(p.m-1)+1
    p.z[2] = ((s[2][1]-p.lower_mu)/(p.upper_mu-p.lower_mu))*(p.m-1)+1
    p.z[3] = (s[3][1,1]/(p.upper_sigma))*(p.m-1)+1
    
    return p.interpolation(p.z[1], p.z[2], p.z[3])
end 
    
function (p::V)(z,s) 
        

    z[1] = ((s[1][1]-p.lower_x)/(p.upper_x-p.lower_x))*(p.m-1)+1
    z[2] = ((s[2][1]-p.lower_mu)/(p.upper_mu-p.lower_mu))*(p.m-1)+1
    z[3] = (s[3][1,1]/(p.upper_sigma))*(p.m-1)+1
    
    return p.interpolation(z[1], z[2], z[3])
end 
    
    
function update!(interp::V)
    itp = interpolate(reshape(interp.values,interp.m,interp.m,interp.m),BSpline(Cubic(Line(OnCell()))))
    itp = extrapolate(itp, Line())
    interp.interpolation = itp
end
    






### compute and store beleif state transitions ####


mutable struct Transitions{T}
    quads
    weights::AbstractVector{Float64}
    Filters::AbstractVector{}
    stateDims::Int64
    actionDims::Int64
    mQuad::Int64
    nodes::AbstractVector{T}
    actions::AbstractVector{}
    values::AbstractArray{T}
    rewards::AbstractArray{}
end 


function init_Transitions(actions,V;mQuad=6)
    

    Filters = []
    quads = []
    for i in 1:Threads.nthreads()
        push!(quads,MvGaussHermite.init_mutable(mQuad,[0.0],[1.0;;])) 
        push!(Filters,init(mQuad))  
    end 
    
    n = length(Filters[1].states)
    
    stateDims = length(V.nodes)
    actionDims = length(actions)
    transitions = Array{typeof(V.nodes[1])}(undef, stateDims, actionDims,n)
    rewards = zeros(stateDims, actionDims)
    weights = Filters[1].quady.weights
                
    return Transitions{typeof(V.nodes[1])}(quads,weights,Filters,stateDims,actionDims,mQuad,V.nodes,actions,transitions,rewards)

end
 

function compute_Transitions!(transitions::Transitions,pars)
    
    i = transitions.stateDims
    j = transitions.actionDims

    acc = zeros(Threads.nthreads())
    total = i*j/Threads.nthreads()
    # 
    Threads.@threads for (i,j) in reshape(collect(Iterators.product(1:i,1:j)),i*j)
        
        id = Threads.threadid()
        if (acc[1] == 0) && (id == 1)
            tick()
        end
        
        acc[id] += 1
        
        if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
            println("progress = ", round(100*acc[1]/total, digits = 1), "%  ", "time: ", round(peektimer(), digits = 0))
                
        end 
        
     
        
        state_transition!(transitions.Filters[id],transitions.nodes[i],transitions.actions[j],pars)
        
        for n in 1:transitions.mQuad
            transitions.values[i,j,n] = deepcopy(transitions.Filters[id].states[n])
        end 

    end 
end 
    
function computeRewards!(transitions,pars)
    
    i = transitions.stateDims
    j = transitions.actionDims
    acc = zeros(Threads.nthreads())
    total = i*j/Threads.nthreads()
    # 
    Threads.@threads for (stateInd,actionInd) in reshape(collect(Iterators.product(1:i,1:j)),i*j)
            
        id = Threads.threadid()

                
        acc[id] += 1
        
        if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
            println("progress = ", round(100*acc[1]/total, digits = 1), "%")
                
        end 
            

        MvGaussHermite.update!(transitions.quads[id],transitions.nodes[stateInd][2],transitions.nodes[stateInd][3])  
        xobs = transitions.nodes[stateInd][1][1]
        transitions.rewards[stateInd,actionInd] = sum(broadcast(x -> R([xobs ,x[1]],transitions.actions[actionInd],pars), 
                                                                            transitions.quads[id].nodes).* transitions.quads[id].weights)
            
    end 
end  
    



### MDP solver ###

function value_expectation!(z,actionInd,stateInd,grid,V)
    sum(broadcast( x->V(z,x), grid.values[stateInd,actionInd,:]).*grid.weights)
end


function bellman!(z,stateInd,grid,V,delta)
    vals =broadcast(i->grid.rewards[stateInd,i]+delta*value_expectation!(z,i,stateInd,grid,V), 1:grid.actionDims)
    return vals[argmax(vals)]
end 

function policy!(z,stateInd,grid,V,delta)
    vals =broadcast(i->grid.rewards[stateInd,i]+delta*value_expectation!(z,i,stateInd,grid,V), 1:grid.actionDims)
    return grid.actions[argmax(vals)]
end

function value_expectation!(z,intermidiate,quad,state,Ht,V,delta,pars)
    state_transition!(intermidiate,state,Ht,pars)
    value =delta*sum(broadcast( x->V(z,x), intermidiate.states).*intermidiate.quady.weights)

    MvGaussHermite.update!(quad,state[2],state[3])  
    xobs = state[1][1]
    reward = sum(broadcast(x -> R([xobs ,x[1]],Ht,pars), quad.nodes).*quad.weights)
    
    return reward+value
end 

function policy!(z,intermidiate,quad,state,actions,V,delta,pars)
    vals = broadcast(Ht -> value_expectation!(z,intermidiate,quad,state,Ht,V,delta,pars),actions) 
    return actions[argmax(vals)]
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

        update!(V)
        test = acc
    end 
end 



end # module 