library(caret)
library(ggplot2)
library(dplyr)
library(dplyr)
library(randomForest)
dat <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/VoI.csv")

names(dat) <- c("sigma_a","sigma_p","SigmaN","Fmsy","NMVmax","price","Bhat","sigma","VoI")

dat <- dat %>% mutate(CV = )
ggplot(dat,aes(x = Bhat, color = sigma, y = VoI))+
  geom_point()+
  viridis::scale_color_viridis()+
  theme_classic()


data_01 <- dat %>%
  mutate(sigma_a = (sigma_a - min(sigma_a))/(max(sigma_a) - min(sigma_a)),
         SigmaN = (SigmaN - min(SigmaN))/(max(SigmaN) - min(SigmaN)),
         Fmsy = (Fmsy - min(Fmsy))/(max(Fmsy) - min(Fmsy)),
         NMVmax = (NMVmax - min(NMVmax))/(max(NMVmax) - min(NMVmax)),
         price = (price - min(price))/(max(price) - min(price)),
         Bhat = (Bhat - min(Bhat))/(max(Bhat) - min(Bhat)),
         sigma = (sigma - min(sigma))/(max(sigma) - min(sigma)),
         VoI = (VoI - min(VoI))/(max(VoI) - min(VoI)))


dat_small<-data_01[sample(1:nrow(dat),100000),]
inTraining <- createDataPartition(dat_small$VoI, p = .8, list = FALSE)
training <- dat_small[ inTraining,]
testing  <- dat_small[-inTraining,]

library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

fitControl <- trainControl(
  ## 5-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated five times
  repeats = 10)

#mtryGrid <- expand.grid(mtry = c(2,3,4,5,6,7,8,9,10))
bstTreeFit1 <- train(VoI ~ ., 
                data = training, 
                method = "ranger",
                trControl = fitControl)
bstTreeFit1


testing$pred <- predict(bstTreeFit1,newdata = testing)


inds <- sample(1:nrow(dat),200000)
training <- dat[inds,]
testing  <- dat[-inds,]
testing <- dat[sample(1:nrow(testing),50000),]
model <- ranger::ranger(VoI~.,data = training,
                        mtry = 7,splitrule = "extratrees",min.node.size=5)

preds <- predict(model, data = testing)
testing$pred <- preds$predictions

ggplot(testing, 
       aes(x = VoI,color=sigma, y = pred))+
  geom_point()+
  geom_abline(aes(slope=1,intercept=0))+
  viridis::scale_color_viridis()+
  theme_classic()

cor(testing$VoI,testing$pred)
testing$error^2 %>% sum()/10000

ggplot(testing %>% mutate(error = (pred - VoI)), 
       aes(x = Bhat,color=sigma, y = error))+
  geom_point()+
  viridis::scale_color_viridis()+
  theme_classic()











data_01 <- dat %>%
  mutate(sigma_a = (sigma_a - min(sigma_a))/(max(sigma_a) - min(sigma_a)),
         SigmaN = (SigmaN - min(SigmaN))/(max(SigmaN) - min(SigmaN)),
         Fmsy = (Fmsy - min(Fmsy))/(max(Fmsy) - min(Fmsy)),
         NMVmax = (NMVmax - min(NMVmax))/(max(NMVmax) - min(NMVmax)),
         price = (price - min(price))/(max(price) - min(price)),
         Bhat = (Bhat - min(Bhat))/(max(Bhat) - min(Bhat)),
         sigma = (sigma - min(sigma))/(max(sigma) - min(sigma)),
         VoI = (VoI - min(VoI))/(max(VoI) - min(VoI)))

inds <- sample(1:nrow(data_01),100000)
train_dat <- data_01[inds,]
test_dat <- data_01[-inds,]
#test_dat <- test_dat[sample(1:nrow(test_dat),10000),]

rf <- randomForest::randomForest(VoI~.,data=train_dat,
                                 ntree = 200)

test_dat$pred<-predict(rf, newdata=test_dat )
test_dat <- test_dat %>% mutate(error = (pred - VoI))

ggplot(test_dat, 
       aes(x = VoI,color=sigma, y = pred))+
  geom_point()+
  geom_abline(aes(slope=1,intercept=0))+
  viridis::scale_color_viridis()+
  theme_classic()
rf
cor(test_dat$VoI,test_dat$pred)
test_dat$error^2 %>% sum()/10000

ggplot(dat_pred %>% mutate(error = (pred - VoI)), 
       aes(x = Bhat,color=sigma, y = error))+
  geom_point()+
  viridis::scale_color_viridis()+
  theme_classic()


