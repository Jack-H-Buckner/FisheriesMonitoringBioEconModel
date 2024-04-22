########## assessment frequency ###############
source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")
file <- "~/github/KalmanFilterPOMDPs/FARM/data/assessment_frequency_pstar.csv"
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
                                buffer = TeX("Unc. buffer $p^*$"),tau=TeX("Growth rate var. $(\\tau)$"),
                                sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
                                H_weight=TeX("Fishery importance $(\\omega_{H})$"),NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
                                c1 =TeX("Stock dep. costs $(c_1)$"),c2 =TeX("Harvest risk ave. $(c_2)$"),
                                NCVshape = TeX("Eco. Imp. risk ave. $(b)$"),discount=TeX("Discount rate $(d)$"))


c("H_weight","Fmsy","tau","sigma_p","buffer","sigma_a","NCV_weight","c1",
  "discount", "c2","NCVshape")

plot_data$variable <- ordered(plot_data$variable,
                              c(H_weight=TeX("Fishery importance $(\\omega_{H})$"),
                                Fmsy=TeX("Productivtiy $(F^{MSY})$"),
                                tau=TeX("Growth rate var. $(\\tau)$"),
                                sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"),
                                buffer = TeX("Unc. buffer $p^*$"),
                                sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
                                NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
                                c1 =TeX("Stock dep. costs $(c_1)$"),
                                c2 =TeX("Harvest risk ave. $(c_2)$"),
                                discount=TeX("Discount rate $(d)$"),
                                NCVshape = TeX("Eco. Imp. risk ave. $(b)$")))

pal = PNWColors::pnw_palette(name="Bay",n=11)


equal_breaks <- function(n = 3, s = 0.05, ...){
  function(x){
    # rescaling
    d <- s * diff(range(x)) / (1+2*s)
    round(seq(min(x)+d, max(x)-d, length=n),
          digits = round(max(-log(max(x),10)+1, 1))
    )
  }
}

p<-ggplot(plot_data,
          aes(x = value,y =inv.logit(yhat),color=variable))+
  geom_line(size=1.5)+
  theme_classic()+
  facet_wrap(~variable,scales = "free_x", 
             labeller = label_parsed,
             nrow = 4 )+
  scale_color_manual(values=pal)+
  scale_x_continuous(breaks=equal_breaks(n=3, s=0.125), 
                     expand = c(0.125, 0.0)) + 
  xlab("Parameter value")+
  ylab("frequency")+
  theme(legend.position="none",
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 18),
        strip.text = element_text(size = 13))

p

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
  scale_y_discrete(labels = c(Fmsy=TeX("Productivity $F_{MSY}$"),
                              buffer=TeX("Unc. buffer $p^*$"),
                              tau=TeX("Env. Var $\\tau$"),
                              sigma_a=TeX("Active monitoring $\\sigma_{a}$"),
                              sigma_p=TeX("Passive monitoring $\\sigma_{p}$"),
                              H_weight=TeX("Fishery imp. $\\omega_{H}$"),
                              NCV_weight=TeX("Ecosystem imp. $\\omega_{NCV}$"),
                              c1=TeX("Stock dep. costs $c_1$"),
                              c2=TeX("nonlinear costs $c_2$"),
                              NCVshape = TeX("Eco. Imp. risk ave. $b$"),
                              discount=TeX("Discount rate $d$")),
                   limits = factor(rev(c("H_weight","Fmsy","tau","sigma_p","buffer","sigma_a","NCV_weight","c1",
                                         "discount", "c2","NCVshape"))))+
  theme_classic()+xlab("Global Sensetivity")+ylab("Parameter")+
  theme(axis.title = element_text(size = 26),axis.text = element_text(size = 22),
        legend.position = c(0.7, 0.5),legend.title= element_text(size = 26),
        legend.text = element_text(size = 22))

p

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/Frequency_variable_importance_new_pstar.png",
       p,height = 7.5, width =7.5)


write.csv(global_sensetivity_data,"~/github/KalmanFilterPOMDPs/examples/data/target_frequency_importance.csv")

