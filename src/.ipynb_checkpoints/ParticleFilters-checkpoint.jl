module ParticleFilters

using StatsBase
using Distributions

mutable struct filter
    N::Int64 # number of samples 
    samples::AbstractVector{AbstractVector{Float64}} 
    weights::AbstractVector{Float64} 
end 

function init(N,d0)
    samples = broadcast(i->rand(d0),1:N) 
    weights = broadcast(i->1/N,1:N)
    return filter(N,samples,weights)
end 


function time_update!(filter,model,a,c)
    filter.samples .= broadcast(i->model.T(filter.samples[i],a,c),1:filter.N)
    filter.samples .= broadcast(i-> rand(Distributions.MvNormal(filter.samples[i],model.SigmaN)),1:filter.N)
    #filter.samples .= broadcast(i->filter.samples[i] .+ rand(d),1:filter.N)
#     s = rand(d,filter.N)
#     for i in 1:filter.N
#         filter.samples[i] .= model.T(filter.samples[i],a,c) + s[:,i]
#     end 
    #filter.samples .= broadcast(i->filter.samples[i] .+ s[:,i],1:filter.N)
#     if any(isnan.(filter.weights))
#         print("tu")
#     end 
    
end 

function measurement_update!(filter,y,model,a,c)
#     loglik = log.(filter.weights) .+ broadcast(i->logpdf(Distributions.MvNormal(filter.samples[i],model.SigmaO(a,c)), y), 1:filter.N)
    
#     loglik = sum(loglik)
    #filter.weights .*= broadcast(i->pdf(Distributions.MvNormal(filter.samples[i],model.SigmaO(a,c)), y), 1:filter.N)
    d = Distributions.MvNormal(zeros(length(filter.samples[1])),model.SigmaO(a,c))
            
    for i in 1:filter.N

        filter.weights[i] *=pdf(d, y .- filter.samples[i])
    end 
    filter.weights .*= 1/sum(filter.weights)
    
    if any(isnan.(filter.weights))
        return "failed"
    else
        return "success"
    end   
end 

# function measurement_update!(filter,y,model,a,c)
#     filter.weights .+= broadcast(i->logpdf(Distributions.MvNormal(filter.samples[i],model.SigmaO(a,c)), y), 1:filter.N)
# end 

function resample!(filter)
    
    if any(isnan.(filter.weights))
        print("rs")
    end 
    
    filter.samples .= wsample(filter.samples, filter.weights, filter.N)
    filter.weights = broadcast(i->1/filter.N,1:filter.N) 
end 

function mean_cov(filter)
    
    mean = zeros(length(filter.samples[1]))
        
    for i in 1:filter.N
        mean .+= filter.samples[i]*filter.weights[i]
    end
    
    # compute covariance 
    Cov_samples = broadcast(i -> (filter.samples[i].-mean) .*
                            (filter.samples[i].- mean)', 
                            1:filter.N)

    Cov = zeros(length(filter.samples[1]),length(filter.samples[1]))
    
    for i in 1:filter.N
        Cov .+= Cov_samples[i]*filter.weights[i]
    end

    return (mean,Cov)
end 
# function mean_cov(filter)
    
#     mean = zeros(length(filter.samples[1]))
#     w = exp.(filter.weights .+ 0.01)./sum(exp.(filter.weights .+ 0.01))
#     for i in 1:filter.N
#         mean .+= filter.samples[i]*w[i]
#     end
    
#     # compute covariance 
#     Cov_samples = broadcast(i -> (filter.samples[i].-mean) .*
#                             (filter.samples[i].- mean)', 
#                             1:filter.N)

#     Cov = zeros(length(filter.samples[1]),length(filter.samples[1]))
    
#     for i in 1:filter.N
#         Cov .+= Cov_samples[i]*w[i]
#     end

#     return (mean,Cov)
# end

end 