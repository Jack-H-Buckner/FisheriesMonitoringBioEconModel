library(dplyr)
library(ggplot2)

file <- "~/github/KalmanFilterPOMDPs/FARM/data/VoI_more_params.csv"
dat <- read.csv(file)
col_names <- c("sigma_a","sigma_p","SigmaN","Fmsy","pstar","NMVmax",
               "bNMV","pi_star","c1","c2","discount","Bhat","CV","VoI")
names(dat) <- col_names

p <- 0.05
N <- nrow(dat)
inds <- sample(1:N,round(N*p))
dat_subset <- dat[inds,]

ggplot(dat_subset ,
       aes(x = pstar,
           y = VoI))+
  geom_point()+
  geom_smooth()



file <- "~/github/KalmanFilterPOMDPs/FARM/data/assessment_frequency_more_params.csv"
dat <- read.csv(file)
col_names <- c("sigma_a","sigma_p","SigmaN","Fmsy",
               "pstar","NMVmax","bNMV","pi_star",
               "c1","c2","discount","frequency")
names(dat) <- col_names


ggplot(dat ,
       aes(x = NMVmax,
           y = frequency))+
  geom_point()+
  geom_smooth()

ggplot(dat ,
       aes(x = NMVmax,
           y = frequency,
           color = paste(pstar < 0.3333,pstar < 0.416333)))+
  geom_smooth()

ggplot(dat ,
       aes(x = NMVmax,
           y = frequency,
           color = paste(bNMV < 0.625)))+
  geom_point()+
  geom_smooth()


ggplot(dat ,
       aes(x = pi_star,
           y = frequency,
           color = paste(c2 < 0.025)))+
  geom_smooth()




