library(dplyr)
library(reshape2)
library(ggplot2)

# histogram data

dat <- read.csv("~/github/KalmanFilterPOMDPs/tests/data/surplus_production_tests_histograms.csv",
                header = FALSE)
names(dat) <- c("mu", "sigma", "kf", "pf")
dat <- reshape2::melt(dat, id.vars = c("mu", "sigma"))

ggplot(dat, aes(x = variable, y = exp(value), fill = variable))+
  geom_violin(alpha = 0.85)+
  facet_grid(mu~sigma)+
  theme_classic()+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay", n=2),
                    name = "Integration Method")+
  xlab("Integration Method: Probability denstiy")+
  ylab("Biomass")+
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 16))

p
ggsave(file = "~/github/KalmanFilterPOMDPs/tests/figures/Tprime_mean_var.png",
       p)


# low sigma 

dat <- read.csv("~/github/KalmanFilterPOMDPs/tests/data/surplus_production_tests_histograms_harvests.csv",
                header = FALSE)
names(dat) <- c("mu", "h", "kf", "pf")
dat <- reshape2::melt(dat, id.vars = c("mu", "h"))

p<-ggplot(dat, aes(x = variable, y = exp(value), fill = variable))+
  geom_violin(alpha = 0.85)+
  facet_grid(mu~h)+
  theme_classic()+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay", n=2),
                    name = "Integration Method")+
  xlab("Integration Method: Probability denstiy")+
  ylab("Biomass")+
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 16))

p
ggsave(file = "~/github/KalmanFilterPOMDPs/tests/figures/Tprime_harvests.png",
       p)

### compare particle filter to quadrature method

dat_pf <- read.csv("~/github/KalmanFilterPOMDPs/tests/data/value_function_int_pf.csv",
                   header = FALSE)

dat_kf <- read.csv("~/github/KalmanFilterPOMDPs/tests/data/value_function_int_kf.csv",
                   header = FALSE)

dat_kf$B <- c(20,40,60,80,100)
dat_pf$B <- c(20,40,60,80,100)

dat_kf <- reshape2::melt(dat_kf, id.var = "B")
dat_pf <- reshape2::melt(dat_pf, id.var = "B")

dat <- dat_kf
dat$value_pf <- dat_pf$value
dat$var <- plyr::mapvalues(dat$variable,
                                 c("V1","V2","V3","V4"),
                                 c(0.05,0.15,0.25,0.35))

p <- ggplot(dat %>% mutate(v =(value - value_pf)/value), 
       aes(x = var, y = B, fill = v))+
  geom_tile()+
  viridis::scale_fill_viridis(name = "Residual")+
  theme_classic()+
  xlab(latex2exp::TeX(r"($\sigma^2$)"))+
  ylab("Expected Biomass")+
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 16))

p
ggsave(file = "~/github/KalmanFilterPOMDPs/tests/figures/VF_integration_pf_kf.png",
       p)



dat <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policies_c3=1.0.csv",
                   header = FALSE)


names(dat) <- c("Policy", "B", "sigma", "NPV")


p<-ggplot(dat, aes(x=as.factor(B),
                   y = as.factor(
                     round(sqrt(exp(sigma)-1), digits=2)),
                   fill=NPV))+
  geom_tile()+
  facet_wrap(~Policy, ncol=2)+
  geom_text(aes(label = as.character(round(NPV,digits = 2))))+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))
  

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_values_c3=1.0.png",
  p, height = 8, width = 9.5)


  
p<-ggplot(dcast(dat,B+sigma~Policy) %>%
  mutate(P1 = round(100*((`1`-`1`) / `1`), digits = 1),
        P2 =round(100*((`2`-`1`) / `1`), digits = 1),
         P3 =round(100*((`3`-`1`) / `1`), digits = 1),
         P4 =round(100*((`4`-`1`) / `1`), digits = 1))%>%
  select(B,sigma,P1,P2,P3,P4) %>%
  melt(id.vars = c("B", "sigma")),
  aes(y=as.factor(B),
      x = as.factor(round(sqrt(exp(sigma)-1), digits=2)),
      fill=value))+
  geom_tile()+
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  geom_text(aes(label = paste(value, " %")))+
  facet_wrap(~variable, ncol=2, scales = "free")+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_difs_c3=1.0.png",
  p, height = 8, width = 9.5)







dat <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policiesc3=2.0.csv",
                header = FALSE)


names(dat) <- c("Policy", "B", "sigma", "NPV")


p<-ggplot(dat, aes(y=as.factor(B),
                   x = as.factor(
                     round(sqrt(exp(sigma)-1), digits=2)),
                   fill=NPV))+
  geom_tile()+
  facet_wrap(~Policy, ncol=2)+
  geom_text(aes(label = as.character(round(NPV,digits = 2))))+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))
p

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_values_c3=2.0.png",
  p, height = 8, width = 9.5)



p<-ggplot(dcast(dat,B+sigma~Policy) %>%
            mutate(P1 = round(100*((`1`-`1`) / `1`), digits = 1),
                   P2 =round(100*((`2`-`1`) / `1`), digits = 1),
                   P3 =round(100*((`3`-`1`) / `1`), digits = 1),
                   P4 =round(100*((`4`-`1`) / `1`), digits = 1))%>%
            select(B,sigma,P1,P2,P3,P4) %>%
            melt(id.vars = c("B", "sigma")),
          aes(y=as.factor(B),
              x = as.factor(round(sqrt(exp(sigma)-1), digits=2)),
              fill=value))+
  geom_tile()+
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  geom_text(aes(label = paste(value, " %")))+
  facet_wrap(~variable, ncol=2, scales = "free")+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))
p
ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_difs_c3=2.0.png",
  p, height = 8, width = 9.5)



  




dat <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policiesc3=0.5.csv",
                header = FALSE)
names(dat) <- c("Policy", "B", "sigma", "NPV")

datP5 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policies_P5_c3=0.5.csv",
                  header = FALSE)
names(datP5) <- c("Policy", "B", "sigma", "NPV")

dat <- rbind(dat,datP5)


p<-ggplot(dat, aes(y=as.factor(B),
                   x = as.factor(
                     round(sqrt(exp(sigma)-1), digits=2)),
                   fill=NPV))+
  geom_tile()+
  facet_wrap(~Policy, ncol=2)+
  geom_text(aes(label = as.character(round(NPV,digits = 2))))+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))
p

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_values_c3=0.5.png",
  p, height = 8, width = 9.5)



p<-ggplot(dcast(dat,B+sigma~Policy) %>%
            mutate(P2 = round(100*((`2`-`1`) / `1`), digits = 1),
                   P3 =round(100*((`3`-`1`) / `1`), digits = 1),
                   P4 =round(100*((`4`-`1`) / `1`), digits = 1),
                   P5 =round(100*((`5`-`1`) / `1`), digits = 1))%>%
            select(B,sigma,P2,P3,P4,P5) %>%
            melt(id.vars = c("B", "sigma")),
          aes(y=as.factor(B),
              x = as.factor(round(sqrt(exp(sigma)-1), digits=2)),
              fill=value))+
  geom_tile()+
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  geom_text(aes(label = paste(value, " %")))+
  facet_wrap(~variable, ncol=2, scales = "free")+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))
p
ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_difs_c3=0.5.png",
  p, height = 8, width = 9.5)







datP5 <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policies_P5_c3=0.5.csv",
                header = FALSE)


names(dat) <- c("Policy", "B", "sigma", "NPV")


p<-ggplot(dat, aes(y=as.factor(B),
                   x = as.factor(
                     round(sqrt(exp(sigma)-1), digits=2)),
                   fill=NPV))+
  geom_tile()+
  geom_text(aes(label = as.character(round(NPV,digits = 2))))+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))
p

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_values_c3=0.5.png",
  p, height = 8, width = 9.5)



p<-ggplot(dcast(dat,B+sigma~Policy) %>%
            mutate(P1 = round(100*((`1`-`1`) / `1`), digits = 1),
                   P2 =round(100*((`2`-`1`) / `1`), digits = 1),
                   P3 =round(100*((`3`-`1`) / `1`), digits = 1),
                   P4 =round(100*((`4`-`1`) / `1`), digits = 1))%>%
            select(B,sigma,P1,P2,P3,P4) %>%
            melt(id.vars = c("B", "sigma")),
          aes(y=as.factor(B),
              x = as.factor(round(sqrt(exp(sigma)-1), digits=2)),
              fill=value))+
  geom_tile()+
  xlab("Uncertianty (CV)")+
  ylab("Median Biomass")+
  geom_text(aes(label = paste(value, " %")))+
  facet_wrap(~variable, ncol=2, scales = "free")+
  viridis::scale_fill_viridis()+
  theme_classic()+ 
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),
        strip.text = element_text(size = 14))

p

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_difs_c3=0.5.png",
  p, height = 8, width = 9.5)



### bar charts

dat <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/alternative_policiesc3=1.5.csv",
                header = FALSE)


names(dat) <- c("Policy", "B", "sigma", "NPV")
dat<-dat%>%mutate(CV = round(sqrt(exp(sigma)-1),digits = 2))


dat$CV <- plyr::mapvalues(dat$CV, c(0.25,0.37,0.5), c("CV = 0.25","CV = 0.37","CV = 0.5"))


p<-ggplot(dat, aes(x=as.factor(B),y = NPV,fill = as.factor(Policy)))+
  geom_bar(stat = "identity",width = 0.5, position = position_dodge(0.7))+
  facet_wrap(~CV,ncol=1)+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay",n=4), 
                    name = "Policy")+
  theme_classic()+ 
  xlab("Expected Biomass")+
  ylab("NPV")+
  theme(axis.title = element_text(size = 24),
        axis.text = element_text(size = 20),
        legend.title = element_text(size = 25),
        legend.text = element_text(size = 20),
        strip.text = element_text(size = 24))

ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_bar_c3=1.5.png",
  p, height = 8, width = 9.5)



p<-ggplot(dcast(dat,CV+B+sigma~Policy,value.var = "NPV") %>%
         mutate(P1 = round(100*((`1`-`1`) / `1`), digits = 1),
                P2 =round(100*((`2`-`1`) / `1`), digits = 1),
                P3 =round(100*((`3`-`1`) / `1`), digits = 1),
                P4 =round(100*((`4`-`1`) / `1`), digits = 1))%>%
         select(B,CV,sigma,P1,P2,P3,P4) %>%
         melt(id.vars = c("B", "sigma","CV")),
       aes(y=value,
           x = as.factor(B),
           fill=variable))+
  geom_bar(stat = "identity",width = 0.5, position = position_dodge(0.7))+
  geom_hline(aes(yintercept = 0.0), linetype = 2)+
  ylab("% NPV")+
  xlab("Expected Biomass")+
  facet_wrap(~CV, ncol = 1)+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay",n=4), 
                    name = "Policy")+
  theme_classic()+ 
  theme(axis.title = element_text(size = 24),
        axis.text = element_text(size = 20),
        legend.title = element_text(size = 24),
        legend.text = element_text(size = 20),
        strip.text = element_text(size = 20))


ggsave(
  file = "~/github/KalmanFilterPOMDPs/examples/figures/fixed_policies_difs_bar_c3=1.5.png",
  p, height = 8, width = 9.5)


