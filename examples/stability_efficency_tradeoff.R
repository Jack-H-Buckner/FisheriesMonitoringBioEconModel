library(ggplot2)
library(dplyr)
library(reshape2)
dat <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/policies.csv") %>%
  filter(CV != 0.0)

ggplot(dat %>% filter(c2 == 0.02, ncv == 0, sigma_a == 0.1,
                      CV %in% c(0.2,0.4,0.6,0.8)),
       aes(x = mean, y = harvest, color = paste(c2,c3)))+
  geom_line()+
  facet_wrap(~CV)+
  scale_color_manual(values = PNWColors::pnw_palette("Bay",n=3),
                     name = "C3")+
  theme_classic()





ggplot(dat %>% filter(c2 == 0.02,ncv == 0.0, sigma_a == 0.1,
                      CV %in% c(0.2,0.4,0.6,0.8)),
       aes(x = exp(log(mean) - 0.5*log(CV^2+1)), y = harvest, color = CV, group = as.character(CV)))+
  geom_line()+
  facet_wrap(~c3, ncol = 2)+
  viridis::scale_color_viridis(option = "cividis")+
  theme_classic()






dat_sim <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/simulations.csv") 
dat_sim <- dat_sim[12:nrow(dat_sim),]

ggplot(dat_sim%>%
  filter(quantile==0.5)%>%
  dcast(sigma_a+C2+C3+ncv ~variable),
  aes(x = H, y = M,color =C2, group = C2))+
  geom_line()+
  geom_point()+
  geom_vline(aes(xintercept = 10), linetype = 2)+
  facet_grid(sigma_a~ncv)+
  viridis::scale_color_viridis()+
  xlim(7,11)+
  theme_bw()


ggplot(dat_sim%>%
         filter(quantile==0.5)%>%
         dcast(sigma_a+C2+C3+ncv ~variable),
       aes(x = R+1, y = M,color =C2, group = C2))+
  geom_line()+
  geom_point()+
  geom_vline(aes(xintercept = 10), linetype = 2)+
  facet_grid(sigma_a~ncv)+
  viridis::scale_color_viridis()+
  theme_bw()






