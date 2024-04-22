library(ggplot2)
library(dplyr)
library(reshape2)

# plot policies 

dat_policies <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/alternative_policies_map.csv")

names(dat_policies) <- c("Bhat","CV","Policy","value")

ggplot(dat_policies,
       aes(x = Bhat, y = CV, fill = as.factor(value)))+
  geom_tile()+
  theme_classic()+
  facet_wrap(~Policy,ncol=1)+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades",n=2),
                    name = "Action")+
  xlab("Expected Biomass")+
  ylab("Uncertainty (CV)")+
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 30),
        legend.text = element_text(size = 20),
        legend.title = element_text(size = 25),
        strip.text = element_text(size = 20))


ggsave("~/github/KalmanFilterPOMDPs/examples/figures/alt_policy.png",
       height = 8.5,
       width = 4.4)



# optimal policy
ggplot(dat_policies %>% filter(Policy == 1),
       aes(x = Bhat, y = CV, fill = as.factor(value)))+
  geom_tile()+
  theme_classic()+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades",n=2),
                    name = "Action")+
  xlab("Expected Biomass")+
  ylab("Uncertainty (CV)")+
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 30),
        legend.text = element_text(size = 20),
        legend.title = element_text(size = 25),
        strip.text = element_text(size = 20))


ggsave("~/github/KalmanFilterPOMDPs/examples/figures/optimal_policy.png",
       height = 6.0,
       width = 6.4)

# Plots results 
dat <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/alt_policies_performance.csv")
dat[1:5,1:5]
dat$policy <- rep(1:4,9)
dat$B0 <- rep(c(rep(15,4),rep(50,4),rep(110,4)),3)
dat$CV <- c(rep(-0.25,12), rep(-0.5,12), rep(-0.75,12))
dat_ <- dat %>% melt(id.vars = c("policy","B0","CV"))

d <- dat_ %>% 
  group_by(policy,B0,CV) %>%
  summarise(V = mean(value),
            Vsd = sd(value)/sqrt(5000)) 

sigma.labs <- c("CV = 0.25", "CV = 0.5", "CV = 0.75")
names(sigma.labs) <- c("-0.25","-0.5","-0.75")


ggplot(d,
       aes(x = as.factor(B0), y = V, fill = as.factor(policy)))+
  geom_bar(stat="identity", width = 0.6,
           position = position_dodge(width = 0.75))+
  # geom_errorbar(aes(ymin = V-2*Vsd, ymax = V+2*Vsd), width = 0.6,
  #          position = position_dodge(width = 0.75))+
  facet_wrap(~CV,ncol = 1,
             labeller = labeller(CV = sigma.labs))+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay",n=4),
                    name = "Policy")+
  theme_classic()+
  xlab("Expected Biomass")+
  ylab("NPV")+
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 30),
        legend.text = element_text(size = 20),
        legend.title = element_text(size = 25),
        strip.text = element_text(size = 25))


ggsave("~/github/KalmanFilterPOMDPs/examples/figures/alt_policy_performance.png",
       height = 8.5,
       width = 10.0)





d2 <- d %>% 
  dcast(B0+CV~policy, value.var = "V") %>%
  mutate(P1 = 100*(`1`-`1`)/`1`,
         P2 = 100*(`2`-`1`)/`1`,
         P3 = 100*(`3`-`1`)/`1`,
         P4 = 100*(`4`-`1`)/`1`) %>%
  melt(id.var = c("B0","CV"))%>%
  filter(variable %in% c("P1","P2","P3","P4"))



ggplot(d2 ,
       aes(x = as.factor(B0), y = value, fill = as.factor(variable)))+
  geom_bar(stat="identity", width = 0.6,
           position = position_dodge(width = 0.75))+
  facet_wrap(~CV,ncol = 1,
             labeller = labeller(CV = sigma.labs))+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay",n=4),
                    name = "Policy")+
  theme_classic()+
  xlab("Expected Biomass")+
  ylab("Reletive Preformance (%NPV)")+
  geom_hline(aes(yintercept = 0.0),linetype = 2, color = "black")+
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 30),
        legend.text = element_text(size = 20),
        legend.title = element_text(size = 25),
        strip.text = element_text(size = 25))



ggsave("~/github/KalmanFilterPOMDPs/examples/figures/alt_policy_rel_performance.png",
       height = 8.5,
       width = 10.0)




sigma.labs <- c("CV = 0.75", "CV = 0.50", "CV = 0.25")
names(sigma.labs) <- c("-0.75","-0.5","-0.25")


dat$upper <- dat$NPV+2*sqrt(dat_var$NPV)/sqrt(500)
dat$lower <- dat$NPV-2*sqrt(dat_var$NPV)/sqrt(500)

d <- dat %>% melt(id.var = c("Bhat","sigma","Policy"))%>%
  reshape2::dcast(Bhat +sigma~Policy+variable ) %>% 
  mutate(`4_diff`=100*(`4_NPV`-`1_NPV`)/`1_NPV`,`3_diff`=100*(`3_NPV`-`1_NPV`)/`1_NPV`,
         `2_diff`=100*(`2_NPV`-`1_NPV`)/`1_NPV`,`1_diff`=100*(`1_NPV`-`1_NPV`)/`1_NPV`,
         
         `4_lower`=100*(`4_lower`-`1_NPV`)/`1_NPV`,`3_lower`=100*(`3_lower`-`1_NPV`)/`1_NPV`,
         `2_lower`=100*(`2_lower`-`1_NPV`)/`1_NPV`,`1_lower`=100*(`1_lower`-`1_NPV`)/`1_NPV`,
         
         `4_upper`=100*(`4_upper`-`1_NPV`)/`1_NPV`,`3_upper`=100*(`3_upper`-`1_NPV`)/`1_NPV`,
         `2_upper`=100*(`2_upper`-`1_NPV`)/`1_NPV`,`1_upper`=100*(`1_upper`-`1_NPV`)/`1_NPV`) %>%
  melt(id.var = c("Bhat","sigma"))%>%
  mutate(CV = round((-1*sqrt(exp(sigma)-1)),2),
         Policy = substr(variable,1,1),
         var = substr(variable,3,10)) %>%
  reshape2::dcast(Bhat +sigma+CV+Policy~var)

d$Bhat <- as.character(d$Bhat)
d$Bhat <- ordered(d$Bhat,c(15,50,100),c(15,50,100))

ggplot(d ,
       aes(x = Bhat, y = diff, fill = as.factor(Policy)))+
  geom_bar(stat="identity", width = 0.6,
           position = position_dodge(width = 0.75))+
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.6,
                position = position_dodge(width = 0.75))+
  facet_wrap(~CV,ncol = 1,
             labeller = labeller(CV = sigma.labs))+
  scale_fill_manual(values = PNWColors::pnw_palette("Bay",n=4),
                    name = "Policy")+
  theme_classic()+
  xlab("Expected Biomass")+
  ylab("Reletive Preformance (%NPV)")+
  geom_hline(aes(yintercept = 0.0),linetype = 2, color = "black")+
  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 30),
        legend.text = element_text(size = 20),
        legend.title = element_text(size = 25),
        strip.text = element_text(size = 25))


ggsave("~/github/KalmanFilterPOMDPs/examples/figures/alt_policy_rel_performance.png",
       height = 8.5,
       width = 10.0)

