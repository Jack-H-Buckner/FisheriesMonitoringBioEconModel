# Plans

## reorganizing BeliefMDP solvers

I want to redisign how the Solver objects are constructed to allow them to be more modular. This would allow differnt methods for function aproximation, numerical integration and aproximate bayesian inferance to be pieced together more easily, hopefully facilitating trouble shooting if one of the base methods fails. 

In general the plan will be to create sets of objects with similar methods for each component of the computation. The big problems that need to be solved are, aproximating the value function, integrating over stochastic state transitions, optimizing the objective function and aproximating beleif state transitions. Several methods are avaiable for each of these steps each of which presents tradeoffs between accuracy and efficeancy and may perform better or worse on specific problems. In the source code I will defined methods that take a common set of arguments and retun a common set of outputs using each of these methods.


### Beleif state transitions 

The beleif state transition occurs in two steps. First, there is a deterministic step propogating uncertianty through the state transition equations. This can be solved in a number of ways including Kalman filters, particle filters, and function aproximaiton. The second step is stochastic and is determined by the observation made at that point in time. This step also requires a numerical method to aproximate the bayesian update that occurs when the observation is made. It also requires a method to integrate over uncertianty in the observation. 

All belief transition functions will have two primary methods: 

1. Simulate
Arguments: the current state defined as a tuple with a vector defining the current mean and a matrix describing the covariance 
Outputs: the stochastically updated state defined in the same way as the state provided as an argument 
   
2. Integrate 
Arguments: The current state defined as a tuple with mean vector and covariance matrix 
Outputs: a list of states defined as before, a list of quadrature weights assocaited with each state. 

Each of these functions will have a method assocaited with differnt numerical techniques. The paramters for eachfor each method will be stored in structs that I will define. These stucts will also provide a place to prealocate memory for various computations. 
