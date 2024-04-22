# load packages
library(ggplot2)
library(dplyr)
library(reshape2)
library(gbm)
library(pdp)
library(latex2exp)
library(sensobol)
set.seed(2101993)

# useful function
in_ball <- function(point,center,radius){
  distance<-sqrt(sum((point-center)^2))
  return(distance < radius)
}

# define functions 
load_VoI_data <- function(file,col_names,
                          p = 1.0){
  # load data and remove values
  dat <- read.csv(file)
  N <- nrow(dat)
  inds <- sample(1:N,round(N*p))
  dat <- dat[inds,]
  
  names(dat) <- col_names
  dat <- dat %>% filter(sigma_p != 0.0)
  
  # compute ranges for each parameter
  ranges <- dat %>% 
      melt(id.var = "VoI")%>%
      group_by(variable)%>%
      summarize(min = min(value),
                max = max(value))

  
  return(list(data = dat, ranges =  ranges))
}

fit_VoI_model <- function(dat,
                          ntree = 1000,
                          interaction.depth = 12,
                          shrinkage = 0.05){

  # rescale each paramter between 0 and 1
  dat$sample <- 1:nrow(dat)
  dat <- dat %>% 
    melt(id.var = c("VoI","sample"))%>%
    group_by(variable)%>%
    mutate(min = min(value),
           max = max(value))%>%
    ungroup()%>%
    mutate(value = (value - min)/(max-min))%>%
    select(-min,-max)%>%
    dcast(sample+VoI~variable,value.var="value")%>%
    select(-sample)
  
  # seperate training and test data sets 
  train_inds <- sample(1:nrow(dat), round(0.9*nrow(dat)))
  training <- dat[train_inds,]
  testing <- dat[-train_inds,]
  # train model 
  model <- gbm(VoI~. ,
             distribution = "gaussian",
             training,
             n.trees = ntree,
             interaction.depth = interaction.depth,
             shrinkage = shrinkage)
  return(list(model=model,training=training,testing=testing))
}


VoI_model_performance <- function(model,testing){
  # plot model performance
  testing$pred <- predict(model,newdata = testing)
  corr_lab <- paste("cor: ", round(cor(testing$VoI,testing$pred),3))
  model_performance <- ggplot(testing, 
         aes(x = VoI, y = pred))+
    geom_point(alpha = 0.5)+
    geom_abline(aes(slope=1,intercept=0))+
    theme_classic()+
    geom_text(aes(x = 0.0, y = 10, 
                  label = corr_lab), 
              size=4)+
    xlab("test data values")+
    ylab("Predicted")
  
  radius <- 0.5
  center1 <- c(0.5,0.5); center2 <- c(0.1,0.5); center3 <- c(0.5,0.1)
  center4 <- c(0.9,0.5); center5 <- c(0.5,0.9); center6 <- c(0.1,0.1)
  center7 <- c(0.9,0.9)
  
  testing <- testing %>% 
    mutate(group = paste(in_ball(c(CV,Bhat),center1,radius),in_ball(c(CV,Bhat),center2,radius),
                         in_ball(c(CV,Bhat),center3,radius),in_ball(c(CV,Bhat),center4,radius),
                         in_ball(c(CV,Bhat),center5,radius),in_ball(c(CV,Bhat),center6,radius),
                         in_ball(c(CV,Bhat),center7,radius)))
  
  model_performance_params_only <- ggplot(testing, 
                                   aes(x = VoI, y = pred))+
    geom_point(alpha = 0.5)+
    geom_abline(aes(slope=1,intercept=0))+
    theme_classic()+
    facet_wrap(~group)+
    geom_text(aes(x = 7, y = 10, 
                  label = corr_lab), 
              size=4)+
    xlab("test data values")+
    ylab("Predicted")
  
  return(list(cor = cor(testing$VoI,testing$pred),
              performance = model_performance, 
              performance_params_only = model_performance_params_only))
}

# dynamics
load_frequency_data <- function(file,col_names){
  # load data and remove values
  dat <- read.csv(file) 
  names(dat) <- col_names
  dat <- dat %>% filter(sigma_p > 0.0)
  
  # compute ranges for each parameter
  ranges <- dat %>% 
    melt(id.var = "frequency")%>%
    group_by(variable)%>%
    summarize(min = min(value),
              max = max(value))
  return(list(data=dat,ranges=ranges))
}

fit_frequency_model <- function(data,
                                ntree = 1000,
                                interaction.depth = 12,
                                shrinkage = 0.05,
                                p = 0.95){
  # rescale each paramter between 0 and 1
  data$sample <- 1:nrow(data)
  data <- data %>% 
    melt(id.var = c("frequency","sample"))%>%
    group_by(variable)%>%
    mutate(min = min(value),
           max = max(value))%>%
    ungroup()%>%
    mutate(value = (value - min)/(max-min))%>%
    select(-min,-max)%>%
    dcast(sample+frequency~variable,value.var="value")%>%
    select(-sample)

  # seperate training and test data sets 
  train_inds <- sample(1:nrow(data), round(p*nrow(data)))
  training <- data[train_inds,]
  testing <- data[-train_inds,]
  
  
  model <- gbm(log((0.999*frequency+0.0005)/(1-(0.999*frequency+0.0005))) ~.,
               distribution = "gaussian",
               training,
               n.trees = ntree,#2000
               interaction.depth = interaction.depth,#20
               shrinkage = shrinkage)#0.01
  
  return(list(model=model,testing=testing,training=training))
}

frequency_model_performance <- function(model,testing){
  vals <- predict(model,newdata = testing)
  testing$pred <- exp(vals)/(1+exp(vals))
  
  corr_lab <- paste("cor: ", round(cor(testing$frequency,testing$pred),3))
  model_performance_plot <- ggplot(testing, 
                                   aes(x = frequency, y = pred))+
    geom_point(alpha = 0.5)+
    geom_abline(aes(slope=1,intercept=0))+
    theme_classic()+
    geom_text(aes(x = 0.5, y = 0.75, 
                  label = corr_lab), 
              size=4)+
    xlab("test data values")+
    ylab("Predicted")
  
  
  return(list(cor = cor(testing$frequency,testing$pred), plot=model_performance_plot))
}





inv.logit <- function(x){exp(x)/(1+exp(x))}
predict_frequency <- function(data){
  pred <- predict(mod,newdata = data)%>%inv.logit()
  return(pred)
}

patial_dependence_data <- function(model,name,ranges,
                                   ntree=1000,
                                   frac_to_build = 0.01){
  range <- ranges[ranges$variable==name,]
  part <- partial(model,pred.var =c(name),n.trees=ntree)%>%
    reshape2::melt(id.var = "yhat")%>%
    mutate(value = value*(range$max-range$min) + range$min)
  return(part)
}


collect_partial_dependence_data <- function(model,names,ranges,
                                            ntree=1000,frac_to_build = 0.01){
  partial_dependence_data <- patial_dependence_data(model,names[1],ranges)
  for(i in 2:length(names)){
    print(names[i])
    results <- patial_dependence_data(model,names[i],ranges)
    partial_dependence_data <- rbind(partial_dependence_data, results)
  }
  
  partial_dependence_data$variable <- ordered(partial_dependence_data$variable,names)
  return( partial_dependence_data )
}


sensetivity_analysis <- function(model,names,#ranges,test_ranges,
                                 N=2^14,k=6,R=10^3,
                                 type = "norm",conf=0.95){
  mat <- sobol_matrices(N = N, params = names)
  y <- predict(model,newdata = as.data.frame(mat))
  ind <- sobol_indices(Y = y, N = N, params = names, boot = TRUE, R = R,
                       type = type, conf = conf)
  
  global_senstivity <- ind$results
  global_senstivity$Metric <- plyr::mapvalues(global_senstivity$sensitivity,
                                              c("Ti","Si"),
                                              c("Total effect", "First order"))
  global_senstivity$parameters <- ordered(global_senstivity$parameters ,
                                          rev(names))
  
  return(global_senstivity)
}



# check ranges

shrink_range <- function(mat,inds,size = 0.9){
  min_ <- rep(0,ncol(mat))
  scale_ <- rep(1.0,ncol(mat))
  min_[inds] <- (1-size)/2
  scale_[inds] <- size
  mat_ <- mat * rep(scale_ ,each=nrow(mat))
  mat_ <- mat_ + rep(min_,each=nrow(mat_))
  return(mat_)
}

compute_index <- function(model,mat,N,R=10^3,
                          type = "norm",conf=0.95){
  y <- predict(model,newdata = as.data.frame(mat))
  ind <- sobol_indices(Y = y, N = N, params = names, boot = FALSE)
  
  global_senstivity <- ind$results
  global_senstivity$Metric <- plyr::mapvalues(global_senstivity$sensitivity,
                                              c("Ti","Si"),
                                              c("Total effect", "First order"))
  global_senstivity$parameters <- ordered(global_senstivity$parameters ,
                                          rev(names))
  return(global_senstivity)
}

range_sensetivity <- function(model,names,N=2^14){
  n = 0
  # compute base value
  mat <- sobol_matrices(N = N, params = names)
  data <- compute_index(model,mat,N)
  data$sample <- n
  inds <- 1:ncol(mat)
  # one element
  for(i in inds){
    n <- n+1
    mat_ <- shrink_range(mat,i,0.8)
    print(ncol(mat_))
    dat <- compute_index(model,mat_,N)
    print(n)
    dat$sample <- n
    data <- rbind(data,dat)
  }
  #two elements
  inds <- t(combn(inds,2))
  for(i in 1:nrow(inds)){
    n <- n+1
    ind <- inds[i,]
    mat_ <- shrink_range(mat,ind,0.8)
    dat <- compute_index(model,mat_,N)
    dat$sample <- n
    data <- rbind(data,dat)
  }
  return(data)
}

