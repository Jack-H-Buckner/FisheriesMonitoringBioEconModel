########## assessment frequency ###############
source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")
file <- "~/github/KalmanFilterPOMDPs/FARM/data/assessment_frequency.csv"
col_names <- c("Fmsy","pstar","tau","sigma_a","sigma_p","H_weight","NCV_weight",
               "c1","c2","NCVshape","discount","frequency")
data <- load_frequency_data(file,col_names)


# if(file.exists("~/github/KalmanFilterPOMDPs/examples/data/frequency_Model.Rdata")){
#   load("~/github/KalmanFilterPOMDPs/examples/data/frequency_Model.Rdata")
#   load("~/github/KalmanFilterPOMDPs/examples/data/frequency_testing.Rdata")
#   load("~/github/KalmanFilterPOMDPs/examples/data/frequency_training.Rdata")
# }else{
model_results <- fit_frequency_model(data$data,ntree = 1000,interaction.depth = 12,shrinkage = 0.025)
model <- model_results$model
testing <- model_results$testing
training <- model_results$training


training <- model_results$training
performance <- frequency_model_performance(model_results$model,
                                           model_results$testing)

performance$plot
ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/frequency_approx_performance.png",
       performance$plot,height  = 9.5, width =9)




plot_data <- collect_partial_dependence_data(model,col_names[1:11],ranges)


levels(plot_data$variable) <- c(
  Fmsy=TeX("Productivtiy $(F^{MSY})$"),
  pstar = TeX("Unc. buffer $(\\p^{*}$)"),
  tau=TeX("Growth rate var. $(\\tau)$"),
  sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
  sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
  H_weight=TeX("Fishery importance $(\\omega_{H})$"),
  NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
  c1 =TeX("Stock dep. costs $(c_1)$"),
  c2 =TeX("Harvest risk ave. $(c_2)$"),
  NCVshape = TeX("Eco. Imp. risk ave. $(b)$"),
  discount=TeX("Discount rate $(d)$"))

plot_data$variable <- ordered(plot_data$variable,
                              c(H_weight=TeX("Fishery importance $(\\omega_{H})$"),
                                Fmsy=TeX("Productivtiy $(F^{MSY})$"),
                                sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
                                pstar = TeX("Unc. buffer $(\\p^{*}$)"),
                                tau=TeX("Growth rate var. $(\\tau)$"),
                                NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
                                sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
                                discount=TeX("Discount rate $(d)$"),
                                c2 =TeX("Harvest risk ave. $(c_2)$"),
                                NCVshape = TeX("Eco. Imp. risk ave. $(b)$"),
                                c1 =TeX("Stock dep. costs $(c_1)$")))

pal = PNWColors::pnw_palette(name="Bay",n=11)


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
            aes(x = value,y =inv.logit(yhat),color=variable))+
  geom_line(size=1.5)+theme_classic()+
  facet_wrap(~variable,scales = "free_x",
             labeller = label_parsed)+
  scale_color_manual(values=pal)+
  scale_x_continuous(breaks=equal_breaks(n=3, s=0.125), 
                     expand = c(0.125, 0.0)) + 
  xlab("Parameter value")+ylab("VoI")+
  theme(legend.position="none",
        axis.title = element_text(size = 30),
        axis.text = element_text(size = 18),
        strip.text = element_text(size = 13))
p

ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/frequency_partial_dependence.png",
       p,height  = 8.0, width =9.0)




global_sensetivity_data <- sensetivity_analysis(model_results$model,
                                                col_names)
global_sensetivity_data %>% filter(Metric == "First order") %>% select(parameters,original)


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
                              discount=TeX("Discount rate $(d)$"),
                              Bhat=TeX("Estimated biomass $(\\hat{B}_{t})$"),
                              CV = TeX("Uncertianty $(CV_t)$")),
                   
                   limits = factor(rev(c("H_weight","Fmsy","sigma_p","pstar",
                                         "tau","NCV_weight","sigma_a","discount",
                                         "c2","NCVshape","c1")))
  )+
  geom_vline(aes(xintercept=0),color="black",lty=2)+
  theme_classic()+
  xlab("Global Sensetivity")+
  ylab("Parameter")+
  theme(axis.title = element_text(size = 30),
        axis.text = element_text(size = 20),
        legend.position = c(0.7, 0.5),
        legend.title= element_text(size = 30),
        legend.text = element_text(size = 20))
p


ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/frequency_sensetivity.png",
       p,height = 8.0, width =8.5)

