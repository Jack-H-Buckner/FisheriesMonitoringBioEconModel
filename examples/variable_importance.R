
source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")


file <- "~/github/KalmanFilterPOMDPs/FARM/data/VoI.csv"
col_names <- c("Fmsy","Buffer","tau","sigma_a",
               "sigma_p","H_weight","NCV_weight",
               "c1","c2","NCVshape","discount","Bhat","CV","VoI")

results <- load_VoI_data(file,col_names,p=1.0)

log(nrow(results$dat),2)
data <- results$dat
nrow(data)
results$ranges


# ggplot(data%>%filter(VoI > 0),aes(y=VoI,x=CV))+
#   geom_point()


ranges <- results$range


results <- fit_VoI_model(data,interaction.depth = 7)
model <- results$model
testing <- results$testing
training <- results$training


plt <- VoI_model_performance(model,results$testing[1:1000,])
plt$performance

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/VoI_approx_performance.png",
       plt$performance,height  = 9.5, width =9)



plot_data <- collect_partial_dependence_data(model,col_names[1:13],ranges)



levels(plot_data$variable) <- c(
  Fmsy=TeX("Productivtiy $(F^{MSY})$"), buffer = TeX("Buffer $(\\beta$)"),
  tau=TeX("Growth rate var. $(\\tau)$"),sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
  sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), H_weight=TeX("Fishery importance $(\\omega_{H})$"),
  NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),c1 =TeX("Stock dep. costs $(c_1)$"),
  c2 =TeX("Harvest risk ave. $(c_2)$"),NCVshape = TeX("Eco. Imp. risk ave. $(b)$"),
  discount=TeX("Discount rate $(d)$"),Bhat=TeX("Estimated B. $(\\hat{B}_{t})$"),
  CV = TeX("Uncertianty $(CV_t)$"))


plot_data$variable <- ordered(plot_data$variable,c(CV = TeX("Uncertianty $(CV_t)$"),Bhat=TeX("Estimated B. $(\\hat{B}_{t})$"),
                                H_weight=TeX("Fishery importance $(\\omega_{H})$"),Fmsy=TeX("Productivtiy $(F^{MSY})$"),
                                c1 =TeX("Stock dep. costs $(c_1)$"),sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"),
                                buffer = TeX("Buffer $(\\beta$)"),NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
                                sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),tau=TeX("Growth rate var. $(\\tau)$"),
                                c2 =TeX("Harvest risk ave. $(c_2)$"),discount=TeX("Discount rate $(d)$"),NCVshape = TeX("Eco. Imp. risk ave. $(b)$")))


pal = PNWColors::pnw_palette(name="Bay",n=13)


library(grid)
# defining the breaks function, 
# s is the scaling factor (cf. multiplicative expand)
equal_breaks <- function(n = 3, s = 0.05, ...){
  function(x){
    # rescaling
    d <- s * diff(range(x)) / (1+2*s)
    round(seq(min(x)+d, max(x)-d, length=n),
          digits = round(max(-log(max(x),10)+1, 1))
          )
  }
}
p <- ggplot(plot_data,
            aes(x = value,y =yhat,color=variable))+
  geom_line(size=1.5)+theme_classic()+
  facet_wrap(~variable,scales = "free_x",
             labeller = label_parsed)+
  scale_color_manual(values=pal)+
  scale_x_continuous(breaks=equal_breaks(n=3, s=0.125), 
                     expand = c(0.125, 0.0)) + 
  xlab("Parameter value")+ylab("VoI")+
  theme(legend.position="none",
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 18),
        strip.text = element_text(size = 13))
p


ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/VoI_partial_dependence_new.png",
       p,height  = 8.0, width =9.0)


names <- c("Fmsy","Buffer","tau","sigma_a","sigma_p","H_weight",
           "NCV_weight","c1","c2","NCVshape","discount","Bhat","CV")
global_sensetivity_data <- sensetivity_analysis(model,names)

global_sensetivity_data %>% 
  filter(Metric == "First order")%>%
  select(parameters,original)


p <- ggplot(global_sensetivity_data,
            aes(x=original,
                y=parameters,
                xmin=0,
                xmax=original,
                color=Metric, 
                group =Metric,
                xend=original))+
  geom_linerange(color="black",position = position_dodge(width = 0.50))+
  geom_point(size=3,position = position_dodge(width = 0.50))+
  scale_color_manual(values=PNWColors::pnw_palette(name="Lake",n=2))+
  scale_y_discrete(labels = c(Fmsy=TeX("Productivtiy $(F^{MSY})$"),
                              Buffer = TeX("Buffer $(\\beta$)"),
                              tau=TeX("Growth rate var. $(\\tau)$"),
                              sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
                              sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
                              H_weight=TeX("Fishery importance $(\\omega_{H})$"),
                              NCV_weight=TeX("Ecosystem importance $(\\omega_{NCV})$"),
                              c1 =TeX("Stock dep. costs $(c_1)$"),
                              c2 =TeX("Harvest risk aversion $(c_2)$"),
                              NCVshape = TeX("Eco. Imp. risk aversion $(b)$"),
                              discount=TeX("Discount rate $(d)$"),
                              Bhat=TeX("Estimated biomass $(\\hat{B}_{t})$"),
                              CV = TeX("Uncertianty $(CV_t)$")),
                   
                   limits = factor(rev(c("CV","Bhat","H_weight","Fmsy",
                                         "c1","sigma_p","Buffer","NCV_weight","sigma_a",
                                         "tau","c2","discount","NCVshape")))
  )+
  geom_vline(aes(xintercept=0),color="black",lty=2)+
  theme_classic()+
  xlab("Global Sensetivity")+
  ylab("Parameter")+
  theme(axis.title = element_text(size = 24),
        axis.text = element_text(size = 18),
        legend.position = c(0.7, 0.5),
        legend.title= element_text(size = 24),
        legend.text = element_text(size = 18))
p


ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/VoI_sensetivity_new.png",
       p,height = 8.0, width =8.5)



#### robustness test for ranges


names <- c("Fmsy","pstar","tau","sigma_a","sigma_p","H_weight",
           "NCV_weight","c1","c2","NCVshape","discount",
           "Bhat","CV")
library(sensobol)
data <- range_sensetivity(model,names,N=2^10)

data$parameters <- ordered(data$parameters,
               levels = rev(c("CV","Bhat","H_weight","pstar",
                              "Fmsy","c1","sigma_p","sigma_a","NCV_weight",
                              "tau","c2","NCVshape","discount")))
p<-ggplot(data ,
          aes(y = parameters,
              x = original,
              fill = sensitivity))+
  geom_boxplot()+
  scale_fill_manual(values=PNWColors::pnw_palette(name="Lake",n=2))+
  scale_y_discrete(labels = c(Fmsy=TeX("$F^{MSY}$"),pstar = TeX("$\\p^{*}$"),SigmaN=TeX("$\\tau$"),
                              sigma_a=TeX("$\\sigma^{a}$"),sigma_p=TeX("$\\sigma^{p}$"),
                              H_weight=TeX("$\\omega_{H}$"),NCV_weight=TeX("$\\omega_{NCV}$"),
                              c1 =TeX("$c_1$"),c2 =TeX("$c_2$"),NCVshape = TeX("$b$"),
                              discount=TeX("$d$"),Bhat=TeX("$\\hat{B}_{t}$"),CV = TeX("$CV_t$")),
                   
                   limits = factor(rev(c("CV","Bhat","H_weight","pstar",
                                         "Fmsy","c1","sigma_p","sigma_a","NCV_weight",
                                         "tau","c2","NCVshape","discount")))
  )+
  xlab("Global Sensetivity")+
  ylab("Parameter")+
  theme_classic()+
  theme(axis.title = element_text(size = 26),
        axis.text = element_text(size = 22),
        legend.position = "bottom",
        legend.title= element_text(size = 22),
        legend.text = element_text(size = 20))

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/VoI_variable_importance_boundries.png",
       p,height = 7.5, width =6)












########## assessment frequency ###############
source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")
file <- "~/github/KalmanFilterPOMDPs/FARM/data/assessment_frequency.csv"
col_names <- c("Fmsy","buffer","tau","sigma_a","sigma_p","H_weight","NCV_weight",
               "c1","c2","NCVshape","discount","frequency")
data <- load_frequency_data(file,col_names)



model_results <- fit_frequency_model(data$data,ntree = 1000,interaction.depth = 12,shrinkage = 0.025,p = 0.9)
model <- model_results$model
testing <- model_results$testing
training <- model_results$training


training <- model_results$training
performance <- frequency_model_performance(model_results$model,model_results$testing)

performance$plot
ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/frequency_approx_performance.png",
       performance$plot,height  = 9.5, width =9)



variable_names <- c("Fmsy","buffer","tau","sigma_a","sigma_p",
                    "H_weight","NCV_weight","c1","c2",
                    "NCVshape","discount")
partial_dependence_data <- collect_partial_dependence_data(
  model_results$model,variable_names,data$ranges)

partial_dependence_data %>%
  group_by(variable)%>%
  summarize(min_ = min(value),max_ = max(value))


global_sensetivity_data <- sensetivity_analysis(model_results$model,
                                                variable_names)

global_sensetivity_data %>% 
  filter(Metric == "First order")%>%
  select(parameters,original)



plot_data <- partial_dependence_data

levels(plot_data$variable) <- c(Fmsy=TeX("Productivtiy $(F^{MSY})$"),
  buffer = TeX("Buffer $(\\beta$)"),tau=TeX("Growth rate var. $(\\tau)$"),
  sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
  H_weight=TeX("Fishery importance $(\\omega_{H})$"),NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
  c1 =TeX("Stock dep. costs $(c_1)$"),c2 =TeX("Harvest risk ave. $(c_2)$"),
  NCVshape = TeX("Eco. imp. risk ave. $(b)$"),discount=TeX("Discount rate $(d)$"))




plot_data$variable <- ordered(plot_data$variable,
                              c(Fmsy=TeX("Productivtiy $(F^{MSY})$"),H_weight=TeX("Fishery importance $(\\omega_{H})$"),
                                sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
                                c1 =TeX("Stock dep. costs $(c_1)$"),buffer = TeX("Buffer $(\\beta$)"),
                                tau=TeX("Growth rate var. $(\\tau)$"),sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
                                c2 =TeX("Harvest risk ave. $(c_2)$"), discount=TeX("Discount rate $(d)$"),
                                NCVshape = TeX("Eco. imp. risk ave. $(b)$")))

pal = PNWColors::pnw_palette(name="Bay",n=11)


count <- 0
breaks_fun <- function(x) {
  count <<- count + 1
  switch(
    count,
    c(0.1, 0.2, 0.3),c(0.1, 0.2, 0.3),
    c(2.5, 7.5, 12.5),c(2.5, 7.5, 12.5),
    c(1,2,3),c(1,2,3),
    c(2.5, 7.5, 12.5),c(2.5, 7.5, 12.5),
    c(0.0,0.2,0.4),c(0.0,0.2,0.4),
    c(0.05,0.15,0.25),c(0.05,0.15,0.25),
    c(0.03,0.06,0.09),c(0.03,0.06,0.09),
    c(0.15,0.3,0.45),c(0.15,0.3,0.45),
    c(0.01, 0.05, 0.09),c(0.01, 0.05, 0.09),
    c(0.01,0.05,0.09),c(0.01,0.05,0.09),
    c(1.3,1.6,1.9),c(1.3,1.6,1.9)
  )
}

count <- 0
p<-ggplot(plot_data,aes(x = value,y =inv.logit(yhat),color=variable))+
  geom_line(size=1.5)+
  facet_wrap(~variable,scales = "free_x", labeller = label_parsed,nrow = 4 )+
  scale_color_manual(values=pal)+
  scale_x_continuous(breaks = breaks_fun) + #breaks = breaks_fun
  xlab("Parameter value")+ylab("Frequency")+
  theme_classic()+
  theme(legend.position="none",
        axis.title = element_text(size = 26),
        axis.text = element_text(size = 18),
        strip.text = element_text(size = 15))

p

count <- 0
ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/Frequency_partial_dependence_new.png",
       p,height = 8.0, width =8.5)





plot_data <- global_sensetivity_data

levels(plot_data$parameters) <-rev(c("Fmsy","buffer","tau","sigma_a","sigma_p","H_weight",
                                 "NCV_weight","c1","c2","NCVshape","discount"))
  
plot_data$variable <- ordered(plot_data$variable,
                      levels = rev(c("H_weight","sigma_p","Fmsy","buffer","sigma_a","tau",
                                      "NCV_weight","c2","discount","c1","NCVshape")))

pal = PNWColors::pnw_palette(name="Bay",n=17)



p<-ggplot(global_sensetivity_data, aes(x=original, y=parameters,xmin=0,xmax=original,
              color=Metric, group =Metric,xend=original),parse = TRUE)+
  geom_linerange(color="black",position = position_dodge(width = 0.50))+
  geom_point(size=3,position = position_dodge(width = 0.50))+
  scale_color_manual(values=PNWColors::pnw_palette(name="Lake",n=2))+
  geom_vline(aes(xintercept=0),color="black",lty=2)+
  scale_y_discrete(labels = c(Fmsy=TeX("Productivity $(F_{MSY})$"),
                              buffer=TeX("Buffer $(\\beta)$"),
                              tau=TeX("Env. Var $(\\tau)$"),
                              sigma_a=TeX("Active monitoring $(\\sigma_{a})$"),
                              sigma_p=TeX("Passive monitoring $(\\sigma_{p})$"),
                              H_weight=TeX("Fishery imp. $(\\omega_{H})$"),
                              NCV_weight=TeX("Ecosystem imp. $(\\omega_{NCV})$"),
                              c1=TeX("Stock dep. costs $(c_1)$"),
                              c2=TeX("nonlinear costs $(c_2)$"),
                              NCVshape = TeX("Eco. imp. risk ave. $(b)$"),
                              discount=TeX("Discount rate $(d)$")),
                   limits = factor(rev(c("Fmsy","H_weight","sigma_p","NCV_weight","c1","buffer",
                                         "tau","discount","sigma_a", "c2","NCVshape"))))+
  theme_classic()+xlab("Effect of parameter on monitoring \n frequency")+ylab("Parameter")+
  theme(axis.title = element_text(size = 24),axis.text = element_text(size = 22),
        legend.position = c(0.7, 0.5),legend.title= element_text(size = 26),
        legend.text = element_text(size = 22))

p

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/Frequency_variable_importance_new.png",
       p,height = 7.5, width =8.5)


write.csv(global_sensetivity_data,"~/github/KalmanFilterPOMDPs/examples/data/target_frequency_importance.csv")








#### robustness test for ranges


names <- c("Fmsy","pstar","tau","sigma_a","sigma_p",
           "H_weight","NCV_weight","c1","c2",
           "NCVshape","discount")
library(sensobol)
data <- range_sensetivity(model_results$model,names,N=2^10)

data$parameters <- ordered(data$parameters,
                           levels = rev(c("H_weight","Fmsy","sigma_p",
                                          "pstar","tau","sigma_a",
                                          "NCV_weight","c2","discount",
                                          "c1","NCVshape")))


names <- c("Fmsy","pstar","tau","sigma_a","sigma_p","H_weight",
           "NCV_weight","c1","c2","NCVshape","discount",
           "Bhat","CV")
global_sensetivity_data <- sensetivity_analysis(model,names)

p <- ggplot(global_sensetivity_data,
            aes(x=original,
                y=parameters,
                xmin=0,
                xmax=original,
                color=Metric, 
                group =Metric,
                xend=original))+
  geom_linerange(color="black",position = position_dodge(width = 0.50))+
  geom_point(size=3,position = position_dodge(width = 0.50))+
  scale_color_manual(values=PNWColors::pnw_palette(name="Lake",n=2))+
  scale_y_discrete(labels = c(Fmsy=TeX("Productivtiy $(F^{MSY})$"),
                              pstar = TeX("Unc. buffer $(\\p^{*}$)"),
                              tau=TeX("Growth rate var. $(\\tau)$"),
                              sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
                              sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
                              H_weight=TeX("Fishery importance $(\\omega_{H})$"),
                              NCV_weight=TeX("Ecosystem importance $(\\omega_{NCV})$"),
                              c1 =TeX("Stock dep. costs $(c_1)$"),
                              c2 =TeX("Harvest risk aversion $(c_2)$"),
                              NCVshape = TeX("Eco. Imp. risk aversion $(b)$"),
                              discount=TeX("Discount rate $(d)$")),
                   
                   limits = factor(rev(c("H_weight","Fmsy","sigma_p","pstar",
                                         "sigma_a","tau","NCV_weight","c1",
                                         "c2","discount","NCVshape")))
  )+
  geom_vline(aes(xintercept=0),color="black",lty=2)+
  theme_classic()+
  xlab("Global Sensetivity")+
  ylab("Parameter")+
  theme(axis.title = element_text(size = 24),
        axis.text = element_text(size = 18),
        legend.position = c(0.7, 0.5),
        legend.title= element_text(size = 24),
        legend.text = element_text(size = 18))
p

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/Frequency_variable_importance_boundries.png",
       p,height = 8.0, width =8.5)

