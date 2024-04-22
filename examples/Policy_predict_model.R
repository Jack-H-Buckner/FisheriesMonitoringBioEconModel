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

# plot policies 

dat_policies <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/alternative_policies_map.csv")

names(dat_policies) <- c("Bhat","CV","Policy","value")

# optimal policy
ggplot(dat_policies %>% filter(Policy == 1),
       aes(x = Bhat, y = CV, fill = as.factor(value)))+
  geom_tile()+
  theme_classic()+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades",n=2),
                    name = "Action")+
  theme_classic()+
  xlab("Expected Biomass")+
  ylab("CV")+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 20))


ggsave("~/github/KalmanFilterPOMDPs/examples/figures/Base_Policy.png",
       height = 4.5,
       width = 6.0)

# load Policy data 
dat1 <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/Policies.csv")
names(dat1) <- c("sigma_a","sigma_p","SigmaN","Fmsy","NMVmax","price","Bhat","CV","P")


dat1 <- dplyr::mutate(dat1,Bhat=Bhat*Fmsy/10)
dat1 <- dat1 %>% filter(P != 0)
dat <- dat1 %>%
  mutate(Fmsy = (Fmsy - min(Fmsy))/(max(Fmsy) - min(Fmsy)),
         SigmaN = (SigmaN - min(SigmaN))/(max(SigmaN) - min(SigmaN)),
         sigma_a = (sigma_a - min(sigma_a))/(max(sigma_a) - min(sigma_a)),
         sigma_p = (sigma_p - min(sigma_p))/(max(sigma_p) - min(sigma_p)),
         NMVmax = (NMVmax - min(NMVmax))/(max(NMVmax) - min(NMVmax)),
         price = (price - min(price))/(max(price) - min(price)),
         Bhat = (Bhat - min(Bhat))/(max(Bhat) - min(Bhat)),
         CV = (CV - min(CV))/(max(CV) - min(CV)))

d_summary <- dat1 %>% 
  reshape2::melt()%>%
  group_by(variable)%>%
  summarize(min = min(value),
            max = max(value))


train_inds <- sample(1:nrow(dat), nrow(dat)/2)
training <- dat[train_inds,]
testing <- dat[-train_inds,]

ntree = 1000
# unscaled 
mod <- gbm((P-1)~.,
           distribution = "bernoulli",
           training,
           n.trees = ntree,
           interaction.depth = 12,
           shrinkage = 0.05)

testing$pred <- predict(mod,newdata = testing)

print(paste("RMSE: ", sqrt(mean(((testing$P-1)-1/(1+exp(-testing$pred)))^2))))

corr_lab <- paste("cor: ", round(cor(testing$P,1/(1+exp(-testing$pred))),3))
ggplot(testing, 
       aes(x = as.factor(P-1), y = 1/(1+exp(-pred))))+
  geom_boxplot()+
  theme_classic()+
  xlab("test data values")+
  ylab("Predicted")


ggplot(testing, 
       aes(x = as.factor(P-1), y = pred))+
  geom_violin()+
  theme_classic()+
  xlab("test data values")+
  ylab("Predicted")
## plot probability of monitoring as a function of belief state
part_dep<- partial(mod,pred.var =c("CV","Bhat"),n.trees=ntree)


dmax = d_summary[d_summary$variable == "Bhat",]
dmax = dmax$max[1]
ggplot(part_dep,
       aes(y=CV,x=dmax*Bhat,fill=1/(1+exp(-yhat))))+
  geom_tile()+
  viridis::scale_fill_viridis(name = "Prob.")+
  theme_classic()+
  xlab("Expected Biomass")+
  ylab("Uncertianty (CV)")+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 20))

ggsave("~/github/KalmanFilterPOMDPs/examples/figures/Policy_sensetivity.png",
       height = 4.5,
       width = 6.0)






ggplot(dat1, aes(x = Bhat, y = CV, color = as.factor(P)))+
  geom_point(alpha = 0.01)




