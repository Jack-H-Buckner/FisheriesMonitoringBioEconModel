
library(ggplot2)
library(dplyr)
### harvest control rule 

r <- 1.43
b <- 0.004
sigma2 <- 0.05
x <- c(0:150)
y <- r*x/(1+b*x)
y_max <- r*x*exp(sqrt(sigma2))/(1+b*x)
y_min <- r*x*exp(-sqrt(sigma2))/(1+b*x)
y_2max <- r*x*exp(2*sqrt(sigma2))/(1+b*x)
y_2min <- r*x*exp(-2*sqrt(sigma2))/(1+b*x)
dat <- data.frame(B_t = x, B_t1 = y,
                  B_t1_min = y_min,B_t1_max = y_max,
                  B_t1_2min = y_2min,B_t1_2max = y_2max)
ggplot(dat,aes(x=B_t,y=B_t1))+
  geom_line()+
  geom_line(aes(y=x), linetype = 2)+
  geom_ribbon(aes(ymin = B_t1_min, ymax = B_t1_max), 
              alpha = 0.1, color = "grey")+
  geom_ribbon(aes(ymin = B_t1_2min, ymax = B_t1_2max), 
              alpha = 0.1, color = "grey")+
  theme_classic()+
  theme(axis.title = element_text(size = 20),
        axis.text = element_text(size = 15))
  