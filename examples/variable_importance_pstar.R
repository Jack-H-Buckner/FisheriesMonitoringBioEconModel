source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")

VoI_taining_data  <- function(){
  file <- "~/github/KalmanFilterPOMDPs/FARM/data/VoI.csv"
  col_names <- c("Fmsy","pstar","tau","sigma_a","sigma_p","H_weight","NCV_weight","c1","c2","NCVshape","discount","Bhat","CV","VoI")
  data <- load_VoI_data(file,col_names,p=1.0)
  dat <- data$data
  range <- data$ranges %>% filter(variable %in% c("pstar", "H_weight"))
  dat$sample <- 1:nrow(dat)
  dat <- dat %>% melt(id.var = c("VoI","sample"))%>%group_by(variable)%>%
    mutate(min = min(value),max = max(value))%>%ungroup()%>%mutate(value = (value - min)/(max-min))%>%
    select(-min,-max)%>%dcast(sample+VoI~variable,value.var="value")%>%select(-sample)
  
  # seperate training and test data sets 
  train_inds <- sample(1:nrow(dat), round(nrow(dat)/8))
  training <- dat[train_inds,]
  return(list(training = training, range=range))
}

frequency_taining_data  <- function(){
  file <- "~/github/KalmanFilterPOMDPs/FARM/data/assessment_frequency.csv"
  col_names <- c("Fmsy","pstar","tau","sigma_a","sigma_p","H_weight","NCV_weight","c1","c2","NCVshape","discount","frequency")
  data <- load_frequency_data(file,col_names)

  range <- data$range %>% filter(variable %in% c("pstar", "H_weight"))
  dat <- data$data
  dat$sample <- 1:nrow(dat)
  dat <- dat %>% melt(id.var = c("frequency","sample"))%>%group_by(variable)%>%
    mutate(min = min(value),max = max(value))%>%ungroup()%>%mutate(value = (value - min)/(max-min))%>%
    select(-min,-max)%>%dcast(sample+frequency~variable,value.var="value")%>%select(-sample)
  
  # seperate training and test data sets 
  train_inds <- sample(1:nrow(dat), round(nrow(dat)/8))
  training <- dat[train_inds,]

  return(list(training = training, range=range))
}


VoI_plot <- function(){
  data <- VoI_taining_data()
  training <- data$training
  range <- data$range
  load("~/github/KalmanFilterPOMDPs/examples/data/VoI_Model.RData")
  part <- partial(model,
                  pred.var = c("pstar","H_weight"),
                  n.trees=ntree) 
  
  p1 <- ggplot(part %>% 
           melt(id.vars = c("pstar", "H_weight"))%>%
           mutate(pstar= pstar*(range$max[1]-range$min[1])+range$min[1],
                  H_weight= H_weight*(range$max[2]-range$min[2])+range$min[2]),
         aes(x = H_weight, y = value, 
             color = pstar, group = as.factor(pstar))
  )+ 
    geom_line()+
    viridis::scale_color_viridis()+
    theme_classic()+
    ylab("VoI")
  ggsave("~/github/KalmanFilterPOMDPs/examples/figures/Hweight_pstar_VoI.png",
         height = 5,width = 6)
  part <- partial(model,
                  pred.var = c("pstar","NCV_weight"),
                  n.trees=ntree) 
  
  
  p2 <- ggplot(part %>% 
           melt(id.vars = c("pstar", "NCV_weight"))%>%
           mutate(pstar= pstar*(range$max[1]-range$min[1])+range$min[1],
                  NCV_weight= NCV_weight*(range$max[2]-range$min[2])+range$min[2]),
         aes(x = NCV_weight, y = value, 
             color = pstar, group = as.factor(pstar))
  )+ 
    geom_line()+
    viridis::scale_color_viridis()+
    theme_classic()+
    ylab("VoI")
  ggsave("~/github/KalmanFilterPOMDPs/examples/figures/NCVweight_pstar_VoI.png",
         height = 5,width = 6)
  return(list(Hplot = p1, NCVplot = p2))
}


frequency_plot <- function(){
  data <- frequency_taining_data()
  training <- data$training
  range <- data$range
  load("~/github/KalmanFilterPOMDPs/examples/data/frequency_Model.RData")
  
  part <- partial(model,pred.var = c("pstar","H_weight"),n.trees=ntree) 
  p1 <- ggplot(part %>% 
                 melt(id.vars = c("pstar", "H_weight"))%>%
                 mutate(pstar= pstar*(range$max[1]-range$min[1])+range$min[1],
                        H_weight= H_weight*(range$max[2]-range$min[2])+range$min[2]),
               aes(x = H_weight, y = inv.logit(value), color = pstar, group = as.factor(pstar)))+ 
    geom_line()+viridis::scale_color_viridis()+
    theme_classic()+ylab("VoI")
  
  ggsave("~/github/KalmanFilterPOMDPs/examples/figures/Hweight_pstar_freqeuncy.png",
         height = 5,width = 6)
  part <- partial(model,
                  pred.var = c("pstar","NCV_weight"),
                  n.trees=ntree) 
  
  
  p2 <- ggplot(part %>% 
                 melt(id.vars = c("pstar", "NCV_weight"))%>%
                 mutate(pstar= pstar*(range$max[1]-range$min[1])+range$min[1],
                        NCV_weight= NCV_weight*(range$max[2]-range$min[2])+range$min[2]),
               aes(x = NCV_weight, y = inv.logit(value), color = pstar, group = as.factor(pstar)))+ 
    geom_line()+viridis::scale_color_viridis()+
    theme_classic()+ylab("VoI")
  
  ggsave("~/github/KalmanFilterPOMDPs/examples/figures/NCVweight_pstar_frequency.png",
         height = 5,width = 6)
  
  return(list(Hplot = p1, NCVplot = p2))
}


plots <- VoI_plot()
plots <- frequency_plot()
