module ThresholdModels

using NLsolve
using DifferentialEquations
using Distributions
using Optim
using TickTock
include("../src/MvGaussHermite.jl")


gamma0(pars) = -prod(pars[1:3])
Bmsy1(pars) = ((pars[2]+pars[3])+sqrt((pars[2]+pars[3])^2-3*pars[2]*pars[3]))/3
Gmsy1(pars) = pars[1]*Bmsy1(pars)*(pars[2]-Bmsy1(pars))*(Bmsy1(pars)-pars[3])
FMSY1(pars) = Gmsy1(pars)/Bmsy1(pars)

function target1(x,gamma0_,MSY_,FMSY_)
    return [gamma0(x),Gmsy1(x),FMSY1(x)] - [gamma0_,MSY_,FMSY_]
end 

dBdt(x,p,theta) = theta[1]*x*(theta[2]-x)*(x-theta[3])+p[1]*x-p[2]*x/(1+theta[4]*x[1])

function Bt(x,E,theta)
    nu = x[2]
    B0 = exp(x[1])
    prob = ODEProblem((x,p,t)->dBdt(x,p,theta),B0,[0.0,1.0],vcat(nu,E))
    sol = solve(prob)
    return sol.u[end]
end 

function derivs(x,p,theta)
    dB = theta[1]*x[1]*(theta[2]-x[1])*(x[1]-theta[3])+p[1]*x[1]-p[2]*x[1]/(1+theta[4]*x[1])
    dC = p[2]*x[1]/(1+theta[4]*x[1])
    return [dB,dC]
end 

function Ht(x,E,theta)
    nu = x[2]
    B0 = exp(x[1])
    prob = ODEProblem((x,p,t)->derivs(x,p,theta),[B0,0.0000001],[0.0,1.0],vcat(nu,E))
    sol = solve(prob)
    return sol[end]
end 

function Bmsy2(x;Bmax = 500)
    sol = optimize(B-> -1* (Bt([log(B),0.0],0.0,vcat(x,[0]))-B),1.0,Bmax)
    return -1*sol.minimum,sol.minimizer
end 

function target2(x,gamma0_,MSY_,FMSY_)
    MSY,BMSY=Bmsy2(x;Bmax = 500)
    return [gamma0(x),MSY,MSY/BMSY] .- [gamma0_,MSY_,FMSY_]
end 

function reparam2(gamma0_,MSY_,FMSY_)
    sol1 = nlsolve(x->target1(x,gamma0_,MSY_,FMSY_),[0.0001,0.0000001,100])
    sol2 = nlsolve(x->target2(x,gamma0_,MSY_,FMSY_),sol1.zero)
    return sol2.zero
end 


function passive_observation(x0,E,theta) 
    val = log(Ht(x0,E,theta)) 
    return [val]
end 
    
    
function R(x,theta,actions,price,beta,c1,c2,b,nmv)       
    B = exp(x[1])
        
    E,Ot = actions
        
    H =Ht(x,E,theta)[2]
    
    return price*H^beta-c1*H^2-c2/Ot+ b*nmv*B/(1+b*B)
end
     
        

        
function T(x,theta)
    nu = x[2]
    B0 = exp(x[1])
    E = exp(x[3])
    prob = ODEProblem((x,p,t)->derivs(x,p,theta),[B0,0.000000],[0.0,1.0],vcat(nu,E))
    sol = solve(prob)
    return log.(sol[end])
end 

function T(B0,E,theta)
    nu = rand(Distributions.Normal(0,sqrt(theta[5])))
    B0 = exp(B0)
    E *= exp(rand(Distributions.Normal(0,sqrt(theta[6]))))
    prob = ODEProblem((x,p,t)->derivs(x,p,theta),[B0,0.000000],[0.0,1.0],vcat(nu,E))
    sol = solve(prob)
    return log(sol[end][1]),sol[end][2]
end


"""
Beleif state transition:

Bhat - log expected biomass
tau2B - variace of log biomass
E - target effort level
epsilon - N(0,1) random variable (sapling Chat)
theta - parameter: [r,a,k,signa2nu,b,sigma2E,sigma2B,sigma2c]
"""
function time_update(Bhat,tau2B,E,theta;m = 10)
    mu = [Bhat,0,log(E)]
    quad = MvGaussHermite.init_mutable(m,mu,[tau2B 0 0; 0 theta[5] 0;0 0 theta[6]])
    
    # propogate samples through state transition 
    samples = broadcast(x->T(x,theta),quad.nodes)
    
    # compute mean 
    mean = zeros(2)
    for i in 1:length(quad.nodes)
        mean .+= samples[i]*quad.weights[i]
    end
    
    # compute covariance 
    samples_Cov = broadcast(i -> (samples[i].-mean) .*(samples[i].-mean)', 
                            1:length(quad.nodes))
    
    Cov = zeros(2,2)
    for i in 1:length(quad.nodes)
        Cov .+= samples_Cov[i]*quad.weights[i]
    end
    
    return mean, Cov
end 


function sampling_distribution(mean,Cov,sigma2B,theta)
    # covariance matrics
    Cov22 = Cov.+ [sigma2B 0.0; 0.0 theta[7]]
    # compute updated variance
    newCov = Cov - Cov*inv(Cov22)*Cov
    # compute mean sampling distribution 
    EBhat = mean[1]
    M = Cov*inv(Cov22)
    VarBhat = sum(M[1,1:2].^2 .* [Cov22[1,1],Cov22[2,2]]) + 2*prod(M[1,1:2])*Cov[1,2]
    return EBhat, VarBhat, newCov[1,1]
end        

function measurement_update(Bt,Ct,mean,Cov,sigma2B,theta)
    # covariance matrics
    Cov22 = Cov.+ [sigma2B 0.0; 0.0 theta[7]]
    # compute updated variance
    newCov = Cov - Cov*inv(Cov22)*Cov
    # compute samples
    Bt += rand(Distributions.Normal(0,sqrt(sigma2B)))
    Ct += rand(Distributions.Normal(0,sqrt(theta[7])))
    mean += Cov*inv(Cov22)*([Bt,Ct].-mean)
    return mean[1], newCov[1,1]
end    


mutable struct Transitions2d
    quads
    theta::AbstractVector{Float64}
    weights::AbstractVector{Float64}
    dimsMean::Int64
    dimsVar::Int64
    dimsAct::Int64
    NMC::Int64
    nodesMean::AbstractVector{Float64}
    nodesVar::AbstractVector{Float64}
    actions::AbstractVector{}
    values::AbstractArray{Float64}
end 
        
function init_transitions(theta,V,actions;Nfilter=1000,mQuad=10,NMC=250)   
    quads = []
    for i in 1:Threads.nthreads()
        push!(quads,MvGaussHermite.init_mutable(mQuad,[0.0],[1.0;;]))  
    end 

    meanDims = length(V.nodesMean)
    varDims = length(V.nodesVar)
    actDims = length(actions)
    transition = zeros(meanDims,varDims,actDims,mQuad,2)

    return Transitions2d(quads,theta,quads[1].weights,meanDims,varDims,actDims,mQuad,
                                            V.nodesMean,V.nodesVar,actions,transition)
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
            
        if (acc[1] == 0) && (id == 1)
            tick()
        end
        
        acc[id] += 1
        if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
            println("progress = ", round(100*acc[1]/total, digits = 1), "%  ", "time: ", round(peektimer(), digits = 1))
                
        end 

        
        Cov = [transitions.nodesVar[indVar];;]
        
        mean, Cov = time_update(transitions.nodesMean[indMean],transitions.nodesVar[indVar],
                                transitions.actions[indAct][1],transitions.theta;m = 10)
        
        EBhat, VarBhat, newCov = sampling_distribution(mean,Cov,transitions.actions[indAct][2],transitions.theta)
        
        MvGaussHermite.update!(transitions.quads[id], [EBhat], [VarBhat;;])
        for n in 1:transitions.NMC
            transitions.values[indMean,indVar,indAct,n,1] = transitions.quads[id].nodes[n][1]
            transitions.values[indMean,indVar,indAct,n,2] = newCov
        end 

    end 
end 


        
mutable struct Rewards2d
    R::Function
    quads
    dimsMean::Int64
    dimsVar::Int64
    dimsAct::Int64
    nodesMean::AbstractVector{Float64}
    nodesVar::AbstractVector{Float64}
    actions::AbstractVector{}
    values::AbstractArray{Float64}
end 
    



function init_rewards(R,V,actions;mQuad=25)
    
    quads = []
    for i in 1:Threads.nthreads()
        push!(quads,MvGaussHermite.init_mutable(mQuad,[0.0],[1.0;;])) 
    end 
    meanDims = length(V.nodesMean)
    varDims = length(V.nodesVar)
    actDims = length(actions)
    values = zeros(meanDims,varDims,actDims)

    return Rewards2d(R,quads,meanDims,varDims,actDims,V.nodesMean,V.nodesVar,actions,values)
end
       
    
function simulateRewards!(x,s,a,Rewards)
            

    return Rewards.R(x,a)
                                                                
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
            
        if (acc[1] == 0) && (id == 1)
            tick()
        end
        
        acc[id] += 1
        if (mod(acc[1],round(Int,total/20))) == 0 && (id == 1)
            
            println("progress = ", round(100*acc[1]/total, digits = 1), "%  ", "time: ", round(peektimer(), digits = 1))
                
        end 

            
        MvGaussHermite.update!(Rewards.quads[id],[Rewards.nodesMean[indMean]],[Rewards.nodesVar[indVar];;])  
            
        Rewards.values[indMean,indVar,indAct] = sum(broadcast(x -> Rewards.R(x,Rewards.actions[indAct]), 
                                                                            Rewards.quads[id].nodes).* Rewards.quads[id].weights)
            
    end 
end   
      



end 