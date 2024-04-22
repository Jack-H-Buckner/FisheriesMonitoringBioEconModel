library(ggplot2)
library(dplyr)


setwd("~/github/KalmanFilterPOMDPs/examples")
dat_a <- read.csv("Recovery_anticipated.csv", header = F)
dat_s <- read.csv("Recovery_suprise.csv", header = F)

dat_a$trial <- 1:100
dat_s$trial <- 1:100

dat_a$case<- "Anticipated"
dat_s$case <- "Surprise"

dat <- reshape2::melt(rbind(dat_a,dat_s),  id.var = c("trial", "case"))
dat$variable <- plyr::mapvalues(dat$variable, 
                                c("V1","V2","V3","V4","V5","V6",
                                  "V7","V8","V9","V10",),
                                c(1.0,2.0,3.0,4.0,5.0,
                                  6.0,7.0,8.0,9.0,10.0))

ggplot(dat %>% 
         group_by(variable,case)%>%
         summarize(median = median(value),
                   `10%` = quantile(value,0.1),
                   `90%` = quantile(value,0.9)),
       aes(x = as.numeric(variable),y = median, color = case, group = case))+
  geom_point(size = 2.5)+
  geom_line()+
  theme_classic()+
  ylim(0,60)+
  xlab("Precision of observations")+
  ylab("Expected time to recover (years)")+
  scale_color_manual(values=PNWColors::pnw_palette("Winter",n = 2),name = "")+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        legend.text = element_text(size = 16))

ggsave("resiliance.png",
       height = 5,
       width = 7)


