# load packages
library(caret)
library(ggplot2)
library(dplyr)
library(dplyr)
library(gbm)
library(caret)
library(doParallel)
library(ICEbox)
library(pdp)


load_frequency_data <- function(file,col_names,
                      ntree = 10000,
                      interaction.depth = 24,
                      shrinkage = 0.005){
  
  # load data and remove values
  dat <- read.csv(file) 
  names(dat) <- col_names
  dat <- dat %>% filter(sigma_p > 0.0)
  
  # compute ranges for each parameter
  dat_summary <- dat %>% 
    melt(id.var = "omega")%>%
    group_by(variable)%>%
    summarize(min = min(value),
              max = max(value))
  
  # rescale each paramter between 0 and 1
  dat$sample <- 1:nrow(dat)
  dat <- dat %>% 
    melt(id.var = c("omega","sample"))%>%
    group_by(variable)%>%
    mutate(min = min(value),
           max = max(value))%>%
    ungroup()%>%
    mutate(value = (value - min)/(max-min))%>%
    select(-min,-max)%>%
    dcast(sample+omega~variable,value.var="value")
  

  
  # seperate training and test data sets 

  train_inds <- sample(1:nrow(dat), round(nrow(dat)/2))
  training <- dat[train_inds,]
  testing <- dat[-train_inds,]
  testing <- testing[sample(1:nrow(testing), round(nrow(testing))),]
  
  
  
  # train model 
  mod <- gbm(log((0.99*omega+0.005)/(1-(0.99*omega+0.005))) ~.,
             distribution = "gaussian",
             training,
             n.trees = ntree,
             interaction.depth = interaction.depth,
             shrinkage = shrinkage)
  
  # plot model performance
  vals <- predict(mod,newdata = testing)

  testing$pred <- exp(vals)/(1+exp(vals))
  
  corr_lab <- paste("cor: ", round(cor(testing$omega,testing$pred),3))
  model_performance_plot<-ggplot(testing, 
         aes(x = omega, y = pred))+
    geom_point(alpha = 0.5)+
    geom_abline(aes(slope=1,intercept=0))+
    theme_classic()+
    geom_text(aes(x = 0.5, y = 0.75, 
                  label = corr_lab), 
              size=4)+
    xlab("test data values")+
    ylab("Predicted")
  
  return(list(summary_data=dat_summary,model = mod, performance = model_performance_plot))
}


file <- "~/github/KalmanFilterPOMDPs/examples/data/frequency.csv"
col_names <- c("sigma_a","sigma_p","SigmaN","Fmsy","NMVmax","price","omega")
results <- load_frequency_data(file,col_names)
results$performance