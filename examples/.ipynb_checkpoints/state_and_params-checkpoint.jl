module state_and_params


function convert_harvest(B,H,Fmax)
    Fmax = 0.9
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


pars = [1.427,-0.2383, 0.8]
function Bt(x,Ht,pars)

    B = exp(x[1]+x[2])
    Binfty = exp(x[2])
    r= exp(pars[1]+pars[2]*x[2]) 

    H,B=convert_harvest(B, Ht, pars[3])

    Bprime = r*B- (r-1)*B^2/Binfty

    if Bprime <0
        Bprime = 1.0
    end 

    return [log(Bprime)-x[2], x[2]]
    
end 

H = [1.0 0.0]

function R(x,actions,c1,c2,mV0,maxV,pars)
    Ht = actions
    B = exp(x[1]+x[2])
    H,B=convert_harvest(B, Ht, pars[3])
    return H-c1*H/B-c2*H^2 + mV0*B/(1+mV0/maxV*B)
end


end 