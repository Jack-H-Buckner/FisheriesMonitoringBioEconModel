source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")
file <- "~/github/KalmanFilterPOMDPs/FARM/data/VoI_more_params.csv"

ncol(dat)

col_names <- c("sigma_a","sigma_p","SigmaN","Fmsy","pstar","NMVmax",
               "bNMV","pi_star","c1","c2","discount","Bhat","CV","VoI")
names(dat)


results <- load_VoI_data(file,col_names,p)
training <- results$data
testing <- results$data
variable_names <- c("CV","Bhat","pi_star","pstar","sigma_a","sigma_p","Fmsy","SigmaN","NMVmax",
                    "c1","c2","discount")
partial_dependence_data <- collect_partial_dependence_data(results$model,variable_names,results$summary_data)
global_sensetivity_data <- sensetivity_analysis(results$model,variable_names)


ggsave("~/github/KalmanFilterPOMDPs/examples/figures/VoI_model_performace_more_parameters.png",
       results$performance)

# New facet label names for dose variable
plot_data <- partial_dependence_data$partial_dependence_curves
levels(plot_data$variable) <- c(CV = TeX("$CV_t$"),Bhat=TeX("$\\hat{B}_{t}$"),price=TeX("$p$"),
                                sigma_a=TeX("$\\sigma^{a}$"),sigma_p=TeX("$\\sigma^{p}$"),
                                Ftarget=TeX("$p^{target}$"),Fmsy=TeX("$F^{MSY}$"),
                                SigmaN=TeX("$\\tau$"),NMVmax = TeX("$\\pi_{nc}^{*}$"))
plot_data$variable <- ordered(plot_data$variable,
                              c(TeX("$CV_t$"),TeX("$\\hat{B}_{t}$"),TeX("$p$"),TeX("$F^{MSY}$"),
                                TeX("$\\sigma^{p}$"),TeX("$\\sigma^{a}$"),TeX("$\\pi_{nc}^{*}$"),
                                TeX("$p^{target}$"),TeX("$\\tau$")))

pal = PNWColors::pnw_palette(name="Bay",n=9)

p <- ggplot(plot_data,
            aes(x = value,y =yhat,color=variable))+
  geom_line(size=1.5)+
  theme_classic()+
  facet_wrap(~variable,scales = "free_x",
             labeller = label_parsed)+
  scale_color_manual(values=pal)+
  xlab("Parameter value")+
  ylab("VoI")+
  theme(legend.position="none",
        axis.title = element_text(size = 26),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 20))

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/VoI_partial_dependence.png",
       p,height = 7.5, width =9)



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
  scale_y_discrete(labels = c(CV = TeX("$CV_t$"),Bhat=TeX("$\\hat{B}_{t}$"),
                              price=TeX("$p$"),Fmsy=TeX("$F_{MSY}$"),SigmaN=TeX("$\\tau$"),
                              sigma_p=TeX("$\\sigma_{p}$"),sigma_a=TeX("$\\sigma_{a}$"),
                              NMVmax = TeX("$\\pi_{nc}^{*}$"), Ftarget=TeX("$p_{target}$")),
                   limits = factor(rev(c("CV","Bhat","price","Fmsy","sigma_p",
                                         "sigma_a","NMVmax", "Ftarget","SigmaN"))))+
  geom_vline(aes(xintercept=0),color="black",lty=2)+
  theme_classic()+
  xlab("Global Sensetivity")+
  ylab("Parameter")+
  theme(axis.title = element_text(size = 26),
        axis.text = element_text(size = 22),
        legend.position = c(0.7, 0.5),
        legend.title= element_text(size = 22),
        legend.text = element_text(size = 20))

write.csv(global_sensetivity_data,"~/github/KalmanFilterPOMDPs/examples/data/VoI_importance.csv")
ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/VoI_sensetivity.png",
       p,height = 9.0, width =5)





# Frequency  plots
file<-"~/github/KalmanFilterPOMDPs/FARM/data/assessment_frequency_test.csv"
col_names <- c("sigma_a","sigma_p","SigmaN","Fmsy",
               "pstar","NMVmax","bNMV","pi_star",
               "c1","c2","discount","frequency")
results_frequency <- load_frequency_data(file,col_names)

read.csv(file) %>% nrow()


partial_dependence_data <-collect_partial_dependence_data(results_frequency$model,col_names[1:7],
                                                          results$summary_data,
                                                          ntree=1000,frac_to_build = 0.002)

ggsave("~/github/KalmanFilterPOMDPs/examples/figures/frequency_model_performace.png",
       results_frequency$performance)


plot_data <- partial_dependence_data$partial_dependence_curves
levels(plot_data$variable) <- c(sigma_a=TeX("$\\sigma_{a}$"),sigma_p=TeX("$\\sigma_{p}$"),
                                SigmaN=TeX("$\\tau$"),Fmsy=TeX("$F_{MSY}$"),
                                Ftarget=TeX("$p_{target}$"),NMVmax = TeX("$\\pi_{nc}^{*}$"),
                                price=TeX("$p$"))
plot_data$variable <- factor(plot_data$variable, 
                             levels=c(price=TeX("$p$"),Fmsy=TeX("$F_{MSY}$"),
                                      SigmaN=TeX("$\\tau$"),sigma_p=TeX("$\\sigma_{p}$"),
                                      sigma_a=TeX("$\\sigma_{a}$"),NMVmax = TeX("$\\pi_{nc}^{*}$"),
                                      Ftarget=TeX("$p_{target}$")))
pal = PNWColors::pnw_palette(name="Bay",n=7)
p<-ggplot(plot_data,
          aes(x = value,y =inv.logit(yhat),color=variable))+
  geom_line(size=1.5)+
  theme_classic()+
  facet_wrap(~variable,scales = "free_x",
             labeller = label_parsed)+
  scale_color_manual(values=pal)+
  xlab("Parameter value")+
  ylab("frequency")+
  theme(legend.position="none",
        axis.title = element_text(size = 26),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 20))

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/Frequency_partial_dependence.png",
       p,height = 7.5, width =9)

global_sensetivity_data <- sensetivity_analysis(results_frequency$model,col_names[1:7])

plot_data <- global_sensetivity_data
plot_data$parameters <- factor(plot_data$parameters,
                               rev(c("sigma_a","sigma_p","SigmaN","Fmsy",
                                     "pstar","NMVmax","bNMV","pi_star",
                                     "c1","c2","discount","frequency")))
pal = PNWColors::pnw_palette(name="Bay",n=7)
p<-ggplot(global_sensetivity_data,
          aes(x=original,
              y=parameters,
              xmin=0,
              xmax=original,
              color=Metric, 
              group =Metric,
              xend=original),parse = TRUE)+
  geom_linerange(color="black",position = position_dodge(width = 0.50))+
  geom_point(size=3,position = position_dodge(width = 0.50))+
  scale_color_manual(values=PNWColors::pnw_palette(name="Lake",n=2))+
  geom_vline(aes(xintercept=0),color="black",lty=2)+
  scale_y_discrete(labels = c(price=TeX("$p$"),Fmsy=TeX("$F^{MSY}$"),SigmaN=TeX("$\\tau$"),
                              sigma_p=TeX("$\\sigma^{p}$"),sigma_a=TeX("$\\sigma^{a}$"),
                              NMVmax = TeX("$\\pi_{nc}^{*}$"), Ftarget=TeX("$p^{target}$")),
                   limits = factor(rev(c("price","Fmsy","SigmaN","sigma_p","sigma_a",
                                         "NMVmax", "Ftarget"))))+
  theme_classic()+
  xlab("Global Sensetivity")+
  ylab("Parameter")+
  theme(axis.title = element_text(size = 26),
        axis.text = element_text(size = 22),
        legend.position = "bottom",
        legend.title= element_text(size = 22),
        legend.text = element_text(size = 20))

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/Frequency_variable_importance.png",
       p,height = 7.5, width =6)


write.csv(global_sensetivity_data,"~/github/KalmanFilterPOMDPs/examples/data/target_frequency_importance.csv")
