dat_sim <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/simulated_values.csv")
dat_vfi <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/value_function.csv")

dat <- cbind(dat_sim,dat_vfi)
names(dat) <- c("simulated", "VFI")
dat$CV <- factor(c(0.25,0.25,0.25,0.5,0.5,0.5,0.75,0.75,0.75))
dat$Bhat <- factor(c(15,50,110,15,50,110,15,50,110))
library(ggplot2)
ggplot(dat,aes(x = simulated, y = VFI,
               color = CV, shape = Bhat))+
  geom_point(size = 2.0)+
  geom_abline(aes(intercept = 0, slope = 1))+
  theme_classic()+
  scale_color_manual(values = PNWColors::pnw_palette("Bay", n = 3))+
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 12))

ggsave("~/github/KalmanFilterPOMDPs/examples/figures/VFIvsSims.png",
       height = 5.0, width = 6.0)