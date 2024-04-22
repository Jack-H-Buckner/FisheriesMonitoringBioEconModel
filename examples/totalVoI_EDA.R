library(dplyr)
library(ggplot2)
dat <- read.csv("~/github/KalmanFilterPOMDPs/FARM/data/total_VoI.csv")
names(dat) <- c("Fmsy","buffer","tau","sigma_a","sigma_p","H_weight",
                "NCV_weight","c1","c2","b","discount","Bhat/Bmsy",
                "CV","value")


  
source("~/github/KalmanFilterPOMDPs/examples/VoI_predict_model2.R")

file <- "~/github/KalmanFilterPOMDPs/FARM/data/total_VoI.csv"
col_names <- c("Fmsy",
               "buffer",
               "tau",
               "sigma_a",
               "sigma_p",
               "H_weight",
               "NCV_weight",
               "c1",
               "c2",
               "NCVshape",
               "discount",
               "Bhat","CV","VoI")

# define functions 
load_VoI_data <- function(file,col_names,p = 1.0){
  # load data and remove values
  dat <- read.csv(file)
  N <- nrow(dat)
  inds <- sample(1:N,round(N*p))
  dat <- dat[inds,]
  
  names(dat) <- col_names
  dat <- dat %>% filter(sigma_p != 0.0)
  dat <- dat %>% group_by(Fmsy,buffer,tau,sigma_a,sigma_p,
                          H_weight,NCV_weight,c1,c2,
                          NCVshape,discount)%>%
    mutate(max_ = max(abs(VoI)))%>%
    filter(max_ < 10^3)%>%
    select(-max_)
  
  # compute ranges for each parameter
  ranges <- dat %>% 
    melt(id.var = "VoI")%>%
    group_by(variable)%>%
    summarize(min = min(value),
              max = max(value))
  
  
  return(list(data = dat, ranges =  ranges))
}


results <- load_VoI_data(file,col_names,p=1.0)
log(nrow(results$dat),2)
data <- results$dat
nrow(data)
results$ranges

ranges <- results$range


# fit model 
results <- fit_VoI_model(data)
model <- results$model
testing <- results$testing
training <- results$training

# check model 
plt <- VoI_model_performance(model,results$testing[1:1000,])
plt$performance



# plots

plot_data <- collect_partial_dependence_data(model,col_names[1:13],ranges)

col_names <- c("Fmsy","buffer","tau","sigma_a","sigma_p","H_weight",
               "NCV_weight","c1","c2","NCVshape","discount",
               "Bhat","CV","VoI")

levels(plot_data$variable) <- c(
  Fmsy=TeX("Productivtiy $(F^{MSY})$"),
  buffer = TeX("Buffer $(\\beta$)"),
  tau=TeX("Growth rate var. $(\\tau)$"),
  sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
  sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
  H_weight=TeX("Fishery importance $(\\omega_{H})$"),
  NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
  c1 =TeX("Stock dep. costs $(c_1)$"),
  c2 =TeX("Harvest risk ave. $(c_2)$"),
  NCVshape = TeX("Eco. Imp. risk ave. $(b)$"),
  discount=TeX("Discount rate $(d)$"),
  Bhat=TeX("Estimated B. $(\\hat{B}_{t})$"),
  CV = TeX("Uncertianty $(CV_t)$"))

c("discount","H_weight","Fmsy","c1",
  "sigma_p","NCV_weight","buffer","sigma_a",
  "tau","Bhat","CV"
  ,"c2","NCVshape")

plot_data$variable <- ordered(plot_data$variable,
                              c(discount=TeX("Discount rate $(d)$"),
                                H_weight=TeX("Fishery importance $(\\omega_{H})$"),
                                Fmsy=TeX("Productivtiy $(F^{MSY})$"),
                                c1 =TeX("Stock dep. costs $(c_1)$"),
                                sigma_p=TeX("Passive obs. error $(\\sigma^{p})$"), 
                                NCV_weight=TeX("Eco. importance $(\\omega_{NCV})$"),
                                buffer = TeX("Buffer $(\\beta$)"),
                                sigma_a=TeX("Active obs. error $(\\sigma^{a})$"),
                                tau=TeX("Growth rate var. $(\\tau)$"),
                                Bhat=TeX("Estimated B. $(\\hat{B}_{t})$"),
                                CV = TeX("Uncertianty $(CV_t)$"),
                                c2 =TeX("Harvest risk ave. $(c_2)$"),
                                NCVshape = TeX("Eco. Imp. risk ave. $(b)$")))

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


ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/total_VoI_partial_dependence.png",
       p,height  = 8.0, width =9.0)

names <- c("Fmsy","buffer","tau","sigma_a","sigma_p","H_weight",
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
                              buffer = TeX("Buffer $(\\beta$)"),
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
                   
                   limits = factor(rev(c("discount","H_weight","Fmsy","c1",
                                         "sigma_p","NCV_weight","buffer","sigma_a",
                                         "tau","Bhat","CV"
                                         ,"c2","NCVshape")))
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


ggsave(file ="~/github/KalmanFilterPOMDPs/examples/figures/total_VoI_sensetivity.png",
       p,height = 8.0, width =8.5)

