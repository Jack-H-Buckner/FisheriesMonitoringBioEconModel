library(dplyr)
annual_priorities <- list(
  fishery_importance = 15.0,
  ecosystem_importance = 10.0,
  stock_status = 6.0,
  relative_fishing_mortaltiy = 5.0,
  years_assesment_overdue = 10.0,
  assessment_level = 0.0,
  change_in_stock_indicators = 10.0,
  quality_of_stock_indicators = 0.0,
  mean_age_in_catch = 0.0,
  recruitment_variability = 0.0,
  consituent_demand = 5.0
)
annual_priorities <- as.data.frame(annual_priorities) %>%
  reshape2::melt()

target_freqeuncy <- list(
  fishery_importance = 1.0,
  ecosystem_importance = 1.0,
  stock_status = 0.0,
  relative_fishing_mortaltiy = 0.0,
  years_assesment_overdue = 0.0,
  assessment_level = 0.0,
  change_in_stock_indicators = 0.0,
  quality_of_stock_indicators = 0.0,
  mean_age_in_catch = 2.0,
  productivity = 0.0,
  recruitment_variability = 1.0,
  consituent_demand = 0.0
)
target_freqeuncy<- as.data.frame(target_freqeucny) %>%
  reshape2::melt()

# load model frequency data
modeled_frequency <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/target_frequency_importance.csv") %>%
  dplyr::filter(Metric == "First order")%>%
  select(parameters, original)

model_to_frame_work_mapping <- list(
  model = c(
    "CV",
    "Bhat",
    "price",
    "sigma_a",
    "sigma_p",
    "Ftarget",
    "Fmsy",
    "SigmaN",
    "NMVmax",
    "None",
    "None"),
  framework = c(
    "years_assesment_overdue",
    "stock_status",
    "fishery_importance",
    "assessment_level",
    "quality_of_stock_indicators",
    "relative_fishing_mortaltiy",
    "mean_age_in_catch",
    "recruitment_variability",
    "ecosystem_importance",
    "change_in_stock_indicators",
    "consituent_demand"
  ))

modeled_frequency$variable<- plyr::mapvalues(
  modeled_frequency$parameters,
  model_to_frame_work_mapping$model,
  model_to_frame_work_mapping$framework)


frqeuncy_data <- merge(target_freqeucny,modeled_frequency, all = TRUE)
frqeuncy_data$original[is.na(frqeuncy_data$original)] <- 0
frqeuncy_data$framework <- frqeuncy_data$value
frqeuncy_data$model <- frqeuncy_data$original
library(ggplot2)
ggplot(frqeuncy_data, aes(x = framework, y = sqrt(model)))+
  geom_smooth(method = lm,se=F,color = "grey", linetype=2)+
  geom_point()+
  theme_classic()+
  ylab("Model effect size")+
  xlab("Framework weight")+
  theme(
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 20)
  )

ggsave(file = "~/github/KalmanFilterPOMDPs/examples/figures/frequency_comparison.png",
       height = 4.5, width =6)

ggplot(frqeuncy_data, aes(x = framework, y = sqrt(model),
                          label = variable))+
  geom_smooth(method = lm,se=F,color = "grey",
              linetype=2)+
  geom_point()+geom_text(hjust = 0)+
  theme_classic()+
  ylab("Model effect size")+
  xlab("Framework weight")+
  theme(
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 20))


# load model frequency data
modeled_VoI <- read.csv("~/github/KalmanFilterPOMDPs/examples/data/VoI_importance.csv") %>%
  dplyr::filter(Metric == "First order")%>%
  select(parameters, original)

modeled_VoI$variable<- plyr::mapvalues(
  modeled_VoI$parameters,
  model_to_frame_work_mapping$model,
  model_to_frame_work_mapping$framework)



annual_priorities_data <- merge(annual_priorities,modeled_VoI, all = TRUE)
annual_priorities_data$original[is.na(annual_priorities_data$original)] <- 0
annual_priorities_data$framework <- annual_priorities_data$value
annual_priorities_data$model <- annual_priorities_data$original
annual_priorities_data$sign <- c(1.0,1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)

library(ggplot2)
ggplot(annual_priorities_data, 
       aes(x = framework, y = sqrt(model),
           label = variable, color = as.factor(sign)))+
  geom_smooth(method = lm,se=F,color = "grey", linetype=2)+
  geom_point()+
  scale_color_manual(values = c("red","black"))+
  theme_classic()+
  ylab("Model effect size")+
  xlab("Framework weight")+
  theme(
    legend.position  = "none",
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 20))

ggsave(file = "~/github/KalmanFilterPOMDPs/examples/figures/annual_priorities_comparison.png",
       height = 4.5, width = 6)

ggplot(annual_priorities_data, 
       aes(x = framework, y = sign * sqrt(model),
           label = variable))+
  geom_smooth(method = lm,se=F,color = "grey", linetype=2)+
  geom_text(hjust = -0.05)+
  geom_point()+
  theme_classic()+
  ylab("Model effect size")+
  xlab("Framework weight")+
  xlim(0,20)+
  theme(
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 20))

annual_priorities_data$in_both <- c(1,1,1,1,1,0,0,0,1,1,0)
ggplot(annual_priorities_data, 
       aes(x = framework, y = sign * sqrt(model),
           color = as.factor(in_both),
           group = in_both))+
  geom_smooth(method = lm,se=F, linetype=2)+
  geom_point()+
  theme_classic()+
  ylab("Model effect size")+
  xlab("Framework weight")+
  xlim(0,20)+
  theme(
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 20))
