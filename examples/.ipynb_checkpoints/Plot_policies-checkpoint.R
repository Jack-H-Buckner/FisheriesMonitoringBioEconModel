library(ggplot2)
library(dplyr)


edges <- function(img){
  xinds <- c()
  yinds <- c()
  for(i in 1:nrow(img)){
    previous <- 0
    for(j in 1:ncol(img)){
      if(previous != img[i,j]){
        yinds <- append(yinds, i)
        xinds <- append(xinds, j)
        previous = img[i,j]
      }
    }
  }
  return( data.frame(x = xinds, y = yinds))
}

append_inds <- function(img,i,j, xinds1,yinds1,xinds0,yinds0){
  if(img[i,j] == 1){
    xinds1<-append(xinds1, i)
    yinds1<-append(yinds1, i)
  }else{
    xinds0<-append(xinds0, i)
    yinds0<-append(yinds0, i)
  }
}

perimeter <- function(img){
  xinds1 <- c()
  yinds1 <- c()
  xinds0 <- c()
  yinds0 <- c()
  for(n in 1:(2*nrow(img)+ 2*ncol(img)-1) ){
    if(n <= nrow(img)){
      j <- 1
      i <- n
      if(img[i,j] == 1){
        xinds1<-append(xinds1, j)
        yinds1<-append(yinds1, i)
      }
      xinds0<-append(xinds0, j)
      yinds0<-append(yinds0, i)
    } else if(n <= (nrow(img) + ncol(img))){
      j <- n - nrow(img)
      i <- nrow(img)
      if(img[i,j] == 1){
        xinds1<-append(xinds1, j)
        yinds1<-append(yinds1, i)
      }
      xinds0<-append(xinds0, j)
      yinds0<-append(yinds0, i)
    } else if (n < (2 * nrow(img) + ncol(img))){
      j <- ncol(img)
      i <- (2 * nrow(img) + ncol(img)) - n
      if(img[i,j] == 1){
        xinds1<-append(xinds1, j)
        yinds1<-append(yinds1, i)
      }
      xinds0<-append(xinds0, j)
      yinds0<-append(yinds0, i)
    }else{
      j <- (2*nrow(img) +2* ncol(img)) - n
      i <- 1
      if(img[i,j] == 1){
        xinds1<-append(xinds1, j)
        yinds1<-append(yinds1, i)
      }
      xinds0<-append(xinds0, j)
      yinds0<-append(yinds0, i)
    }

  }
  return(list(inds1=data.frame(x = xinds1, y = yinds1), inds0 = data.frame(x = xinds0, y = yinds0)))
}

# Base objective alternative growth
dat2 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/monitoring_P8.csv", header = F)

names(dat2) <- c("E_log_B", "V_log_B", "value")

img <- reshape2::dcast(dat2, E_log_B~V_log_B) %>%
  select(-E_log_B)

dat_p <- perimeter(img)
dat_p$inds0$E_log_B = unique(dat2$E_log_B)[dat_p$inds0$y]
dat_p$inds0$V_log_B = unique(dat2$V_log_B)[dat_p$inds0$x]

dat <- edges(img)
dat <- rbind(dat ,dat_p$inds1)
dat$E_log_B = unique(dat2$E_log_B)[dat$y]
dat$V_log_B = unique(dat2$V_log_B)[dat$x]

p<-ggplot(dat_p$inds0, aes(x=exp(E_log_B +0.5*V_log_B),y = sqrt(exp(V_log_B)-1))) + 
  geom_polygon( aes(fill = "0"))+
  geom_polygon(data = dat, aes(fill = "1" ))+
  theme_classic()+
  xlab("Expected Biomass")+
  ylab("Uncertianty (CV)")+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor") + 
  theme(axis.title = element_text(size = 24),
        axis.text = element_text(size = 20),
        legend.title = element_text(size = 24),
        legend.text = element_text(size = 20))
p
ggsave(file = "~/github/KalmanFilterPOMDPs/examples/figures/Policy_base_40_10.png",
       p, height = 5, width = 7)

# Risk neutralalternative growth 
dat2 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/monitoring_P3.csv", header = F)

names(dat2) <- c("E_log_B", "V_log_B", "value")

img <- reshape2::dcast(dat2, E_log_B~V_log_B) %>%
  select(-E_log_B)

dat_p <- perimeter(img)
dat_p$inds0$E_log_B = unique(dat2$E_log_B)[dat_p$inds0$y]
dat_p$inds0$V_log_B = unique(dat2$V_log_B)[dat_p$inds0$x]

dat <- edges(img)
dat <- rbind(dat ,dat_p$inds1)
dat$E_log_B = unique(dat2$E_log_B)[dat$y]
dat$V_log_B = unique(dat2$V_log_B)[dat$x]

ggplot(dat_p$inds0, aes(y=exp(E_log_B +0.5*V_log_B),x = sqrt(exp(V_log_B)-1))) + 
  geom_polygon( aes(fill = "0"))+
  geom_polygon(data = dat, aes(fill = "1" ))+
  theme_classic()+
  ylab("Expected Biomass")+
  xlab("Uncertianty (CV)")+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor")+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14))




# Strong risk aversion alternative growth 
dat2 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/monitoring_P4.csv", header = F)

names(dat2) <- c("E_log_B", "V_log_B", "value")

img <- reshape2::dcast(dat2, E_log_B~V_log_B) %>%
  select(-E_log_B)

dat_p <- perimeter(img)
dat_p$inds0$E_log_B = unique(dat2$E_log_B)[dat_p$inds0$y]
dat_p$inds0$V_log_B = unique(dat2$V_log_B)[dat_p$inds0$x]

dat <- edges(img)
dat <- rbind(dat ,dat_p$inds1)
dat$E_log_B = unique(dat2$E_log_B)[dat$y]
dat$V_log_B = unique(dat2$V_log_B)[dat$x]

ggplot(dat_p$inds0, aes(y=exp(E_log_B +0.5*V_log_B),x = sqrt(exp(V_log_B)-1))) + 
  geom_polygon( aes(fill = "0"))+
  geom_polygon(data = dat, aes(fill = "1" ))+
  theme_classic()+
  ylab("Expected Biomass")+
  xlab("Uncertianty (CV)")+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor")+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14))






# base objective constant effort p-star
dat2 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/monitoring_P5.csv", header = F)

names(dat2) <- c("E_log_B", "V_log_B", "value")

img <- reshape2::dcast(dat2, E_log_B~V_log_B) %>%
  select(-E_log_B)

dat_p <- perimeter(img)
dat_p$inds0$E_log_B = unique(dat2$E_log_B)[dat_p$inds0$y]
dat_p$inds0$V_log_B = unique(dat2$V_log_B)[dat_p$inds0$x]

dat <- edges(img)
dat <- rbind(dat ,dat_p$inds1)
dat$E_log_B = unique(dat2$E_log_B)[dat$y]
dat$V_log_B = unique(dat2$V_log_B)[dat$x]

ggplot(dat_p$inds0, aes(y=exp(E_log_B +0.5*V_log_B),x = sqrt(exp(V_log_B)-1))) + 
  geom_polygon( aes(fill = "0"))+
  geom_polygon(data = dat, aes(fill = "1" ))+
  theme_classic()+
  ylab("Expected Biomass")+
  xlab("Uncertianty (CV)")+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor")+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14))



# risk averse  constant effort p-star
dat2 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/monitoring_P6.csv", header = F)

names(dat2) <- c("E_log_B", "V_log_B", "value")

img <- reshape2::dcast(dat2, E_log_B~V_log_B) %>%
  select(-E_log_B)

dat_p <- perimeter(img)
dat_p$inds0$E_log_B = unique(dat2$E_log_B)[dat_p$inds0$y]
dat_p$inds0$V_log_B = unique(dat2$V_log_B)[dat_p$inds0$x]

dat <- edges(img)
dat <- rbind(dat ,dat_p$inds1)
dat$E_log_B = unique(dat2$E_log_B)[dat$y]
dat$V_log_B = unique(dat2$V_log_B)[dat$x]

ggplot(dat_p$inds0, aes(y=exp(E_log_B +0.5*V_log_B),x = sqrt(exp(V_log_B)-1))) + 
  geom_polygon( aes(fill = "0"))+
  geom_polygon(data = dat, aes(fill = "1" ))+
  theme_classic()+
  ylab("Expected Biomass")+
  xlab("Uncertianty (CV)")+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor")+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14))






dat_bh <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/biomass_harvest1.csv", header = F)

names(dat_bh) <- c("biomass","harvest","residual")
ggplot(dat_bh, aes(x = biomass, y = harvest))+geom_point()+geom_smooth()


ggplot(dat_bh, aes(x = biomass, y = residual))+geom_point()+geom_smooth()






### plot fixed policies 
perimiter_plot_data <- function(dat,p){
  datP1 <- dat %>% filter(P == p) %>%select(-P)
  imgP1 <- reshape2::dcast(datP1,E_log_B~V_log_B) %>%select(-E_log_B)
  dat_pP1 <- perimeter(imgP1)
  dat_pP1$inds0$E_log_B = unique(datP1$E_log_B)[dat_pP1$inds0$y]
  dat_pP1$inds0$V_log_B = unique(datP1$V_log_B)[dat_pP1$inds0$x]
  
  dat0P1 <- edges(imgP1)
  dat0P1 <- rbind(dat0P1 ,dat_pP1$inds1)
  dat0P1$E_log_B = unique(datP1$E_log_B)[dat0P1$y]
  dat0P1$V_log_B = unique(datP1$V_log_B)[dat0P1$x]
  
  dat0P1$value <- 1
  dat_pP1$inds0$value <- 0

  datP1 <- rbind(dat0P1,dat_pP1$inds0)
  
  datP1$P <- p
  return(datP1)
}

dat <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policies_map.csv", header = F)
names(dat) <- c("E_log_B", "V_log_B", "P", "value")


p <- ggplot(dat, aes(x=exp(log(E_log_B) +0.5*V_log_B),
                  y = sqrt(exp(V_log_B)-1),
                  color = as.factor(value))) + 
  geom_point( )+
  theme_classic()+
  facet_wrap(~P, nrow=2)+
  xlab("Expected Biomass")+
  ylab("Uncertianty (CV)")+
  scale_color_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor")+ 
  theme(axis.title = element_text(size = 36),
        axis.text = element_text(size = 18),
        legend.title = element_text(size =36),
        legend.text = element_text(size = 30),
        strip.text = element_text(size = 30))

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_2cols.png",
  p, height = 8.5, width = 10.5)








