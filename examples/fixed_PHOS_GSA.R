library(randtoolbox)
source("~/github/HatcheriesQuantGenModel/src/GeneticDemographic.R")
source("~/github/HatcheriesQuantGenModel/src/FixedPHOSPolicy.R")

# Define simulation a function of model parameters

# parameters
# PNOB, PHOS, brood_stock, release_mortality, standard_deviation
simulation <- function(test_params, 
                       base_params,
                       simulation_length = 2200,
                       burnin = 200){
  
  # set model and policy with new parameters 
  policy <- init_fixed_PHOS(
    PNOB = test_params$PNOB, 
    PHOS = test_params$PHOS,
    brood_stock = test_params$brood_stock,
    release_mortality = test_params$release_mortality,
    standard_deviation = test_params$standard_deviation
  )
  
  model <- init_model(
    max_age = base_params$max_age,
    natural_mortality = base_params$natural_mortality,
    maturation = base_params$maturation,
    fecunity_at_age =  base_params$fecunity_at_age,
    R0_wild = test_params$R0_wild, 
    R0_hatchery = base_params$R0_hatchery, 
    Seq_wild = base_params$Seq_wild, # equilibrium abundance wild
    competition_hatchery = base_params$competition_hatchery, # equilibrium abundance hatchery
    competition_wild_hatchery = base_params$competition_wild_hatchery, # fraction of with in type competition
    recruitment_sd_wild = test_params$recruitment_sd_wild,
    recruitment_sd_hatchery = test_params$recruitment_sd_hatchery, # recruitment variability
    recruitment_auto_corr = test_params$recruitment_auto_corr,
    recruitment_corr = test_params$recruitment_corr,
    max_collection_wild = base_params$max_collection_wild,
    max_collection_hatchery = base_params$max_collection_hatchery,
    release_mortaltiy = test_params$release_mortaltiy,
    reletive_fitness_ = test_params$reletive_fitness_,
    selection = test_params$selection)
  
  # run simulation
  ## set accumulators to collect time series
  spawners_wild <- c()
  genotype_hathery <- c()
  genotype_wild <- c()
  harvest <- c()
  
  ## run model 
  for(t in 1:simulation_length){
    vals <- time_step_fixed_PHOS(model,policy)
    model <- vals$model
    policy <- vals$policy
    if(t > burnin){
      spawners_wild <- append(spawners_wild, vals$spawners_wild)
      genotype_hathery <- append(genotype_hathery,model@genotype_hatchery)
      genotype_wild <- append(genotype_wild,model@genotype_wild)
      harvest <- append(harvest, vals$harvest)
    }
  }
  
  # compute statistics
  E.spawners_wild <- mean(spawners_wild)
  V.spawners_wild <- sd(spawners_wild)^2
  E.genotype_hathery <- mean(genotype_hathery)
  V.genotype_hathery <- sd(genotype_hathery)^2
  E.genotype_wild <- mean(genotype_wild)
  V.genotype_wild <- sd(genotype_wild)^2
  E.harvest <- mean(harvst)
  V.harvest <- sd(harvst)^2
  
  # data
  dat <- data.frame(
    PNOB = test_params$PNOB, 
    PHOS = test_params$PHOS,
    brood_stock = test_params$brood_stock,
    release_mortality = test_params$release_mortality,
    standard_deviation = test_params$standard_deviation,
    R0_wild = test_params$R0_wild, 
    recruitment_sd_wild = test_params$recruitment_sd_wild,
    recruitment_sd_hatchery = test_params$recruitment_sd_hatchery, 
    recruitment_auto_corr = test_params$recruitment_auto_corr,
    recruitment_corr = test_params$recruitment_corr,
    reletive_fitness_ = test_params$reletive_fitness_,
    selection = test_params$selection,
    E.spawners_wild = E.spawners_wild, 
    V.spawners_wild = V.spawners_wild,
    E.genotype_hathery = E.genotype_hathery,
    V.genotype_hathery = V.genotype_hathery,
    E.genotype_wild = E.genotype_wild,
    V.genotype_wild = V.genotype_wild,
    E.harvest = E.harvest,
    V.harvest = V.harvest
  )
  
  return(dat)
}


# Define ranges for parameter values 
# generate a quasi random set of parameter values 
sample_parameters <- function(upper_bound,
                              lower_bound,
                              num_params,
                              log_2_num_samples){
  lower_bound_params
}
num_params = length(lower_bound_params )
log_2_num_samples = 11
samples <- sobol(2^log_2_num_samples,dim=num_params)
## rescale quasi random numbers to match parameter value ranges












