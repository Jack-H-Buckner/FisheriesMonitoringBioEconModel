library(ggplot2)
library(dplyr)
library(reshape2)
library(latex2exp)
plot_policies <- function(figure_file1,figure_file2,file1,file2,file3,file4,
                          hline = 0.4){
dat_04_0 <- read.csv(file1)
dat_05_0 <- read.csv(file2)
dat_04_1 <- read.csv(file3)
dat_05_1 <- read.csv(file4)

names(dat_04_0) <- c("Bhat","CV","Monitor")
dat_04_0$buffer <- "with uncertanty buffer"
dat_04_0$threshold <- "no reference point"
names(dat_05_0) <- c("Bhat","CV","Monitor")
dat_05_0$buffer <- "no uncertanty buffer"
dat_05_0$threshold <- "no reference point"
names(dat_04_1) <- c("Bhat","CV","Monitor")
dat_04_1$buffer <- "with uncertanty buffer"
dat_04_1$threshold <- "with reference point"
names(dat_05_1) <- c("Bhat","CV","Monitor")
dat_05_1$buffer <- "no uncertanty buffer"
dat_05_1$threshold <- "with reference point"
dat <- rbind(dat_04_0,dat_05_0)%>%
  rbind(dat_04_1)%>%rbind(dat_05_1)

p<-ggplot(dat,
       aes(x = Bhat, y = CV,
           fill = as.factor(Monitor == 2)))+
  geom_tile()+
  facet_grid(buffer~threshold)+
  scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                    name = "Monitor")+
  theme_classic()+
  xlab(TeX("Expected biomass: $\\hat{B}_t$"))+
  ylab(TeX("Uncertainty: $CV_t$"))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18),
        strip.text = element_text(size = 18),
        legend.position = "bottom")

  ggsave(figure_file1, height = 7, width = 7)
  
  p <- ggplot(dat %>%
                filter(Monitor == 2)%>%
                group_by(buffer, threshold, Bhat)%>%
                summarize(CV = min(CV)),
              aes(x = Bhat, y = CV,
                  color = paste(buffer)))+
    geom_line()+
    facet_wrap(~threshold, ncol = 1)+
    scale_color_manual(values = PNWColors::pnw_palette("Bay", n=2),
                      name = "")+
    theme_classic()+
    xlab(TeX("Expected biomass: $\\hat{B}_t$"))+
    ylab(TeX("Uncertainty: $CV_t$"))+
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 18),
          legend.text = element_text(size = 16),
          legend.title = element_text(size = 2),
          strip.text = element_text(size = 18),
          legend.position = "bottom",
          legend.direction = "vertical")
  
  ggsave(figure_file2, height = 7, width = 5)
  return(p)
}


plot_policy <- function(figure_file, data_file){
  
  data <- read.csv(data_file)
  names(data) <- c("Bhat", "CV", "Monitor")
  p<-ggplot(data,aes(x = Bhat, y = CV,fill = as.factor(Monitor == 2)))+
    geom_tile()+
    scale_fill_manual(values = PNWColors::pnw_palette("Cascades", n=2),
                      name = "Monitor")+
    theme_classic()+
    xlab(TeX("Expected Biomass: $\\hat{B}_t$"))+
    ylab(TeX("Uncertainty: $CV_t$"))+
    theme(axis.text = element_text(size = 19),
          axis.title = element_text(size = 22),
          legend.text = element_text(size = 19),
          legend.title = element_text(size = 22),
          strip.text = element_text(size = 18),
          legend.position = "right")
  
  ggsave(figure_file, height = 5, width = 7.5)

  return(p)
}

plot_policy("~/github/KalmanFilterPOMDPs/examples/figures/Base_Policy.png", "/Users/johnbuckner/github/KalmanFilterPOMDPs/FARM/data/base_policy.csv")
  
file1<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.3_Bthreshold_0.0_NCV_0.0_c2_0.0.csv"
file2<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.0_Bthreshold_0.0_NCV_0.0_c2_0.0.csv"
file3<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.3_Bthreshold_1.0_NCV_0.0_c2_0.0.csv"
file4<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.0_Bthreshold_1.0_NCV_0.0_c2_0.0.csv"
figure_file1 <- "~/github/KalmanFilterPOMDPs/examples/figures/Policies.png"
figure_file2 <- "~/github/KalmanFilterPOMDPs/examples/figures/Policies_threshold.png"
plot_policies(figure_file1,figure_file2,file1,file2,file3,file4)




file1<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.3_Bthreshold_0.0_NCV_0.0_c2_0.05.csv"
file2<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.0_Bthreshold_0.0_NCV_0.0_c2_0.05.csv"
file3<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.3_Bthreshold_1.0_NCV_0.0_c2_0.05.csv"
file4<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.0_Bthreshold_1.0_NCV_0.0_c2_0.05.csv"
figure_file1 <- "~/github/KalmanFilterPOMDPs/examples/figures/Policies_1.0_NCV_0.0_c2_0.05.png"
figure_file2 <- "~/github/KalmanFilterPOMDPs/examples/figures/Policies_threshold_1.0_NCV_0.0_c2_0.05.png"
plot_policies(figure_file1,figure_file2,file1,file2,file3,file4)

file1<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.3_Bthreshold_0.0_NCV_10_c2_0.0.csv"
file2<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.0_Bthreshold_0.0_NCV_10_c2_0.0.csv"
file3<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.3_Bthreshold_1.0_NCV_10_c2_0.0.csv"
file4<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_buffer_0.0_Bthreshold_1.0_NCV_10_c2_0.0.csv"
figure_file1 <- "~/github/KalmanFilterPOMDPs/examples/figures/Policies_1.0_NCV_10_c2_0.0.png"
figure_file2 <- "~/github/KalmanFilterPOMDPs/examples/figures/Policies_threshold_1.0_NCV_10_c2_0.0.png"
plot_policies(figure_file1,figure_file2,file1,file2,file3,file4)

library(ggplot2)
library(dplyr)
library(reshape2)
library(latex2exp)
plot_thresholds<- function(file1,file2,file3,file4,file5,file6,
                          hline = 0.4){
  dat_04 <- read.csv(file1)
  dat_05 <- read.csv(file2)
  dat_NCV_04 <- read.csv(file3)
  dat_NCV_05 <- read.csv(file4)
  dat_r_04 <- read.csv(file5)
  dat_r_05 <- read.csv(file6)
  
  names(dat_04) <- c("Bhat","CV","Monitor")
  dat_04$objecive <- "Harvest only"
  dat_04$buffer <- "With uncertianty Buffer"
  names(dat_05) <- c("Bhat","CV","Monitor")
  dat_05$objecive <- "Harvest only"
  dat_05$buffer <- "With out uncertianty Buffer"
  names(dat_NCV_04) <- c("Bhat","CV","Monitor")
  dat_NCV_04$objecive <- "Non-consumptive values"
  dat_NCV_04$buffer <- "With uncertianty Buffer"
  names(dat_NCV_05) <- c("Bhat","CV","Monitor")
  dat_NCV_05$objecive <- "Non-consumptive values"
  dat_NCV_05$buffer <- "With out uncertianty Buffer"
  names(dat_r_04) <- c("Bhat","CV","Monitor")
  dat_r_04$objecive <- "Risk averse"
  dat_r_04$buffer <- "With uncertianty Buffer"
  names(dat_r_05) <- c("Bhat","CV","Monitor")
  dat_r_05$objecive <- "Risk averse"
  dat_r_05$buffer <- "With out uncertianty Buffer"
  
  dat <- rbind(dat_04,dat_05)%>%
    rbind(dat_NCV_04)%>%rbind(dat_NCV_05)%>%
    rbind(dat_r_04)%>%rbind(dat_r_05)
  
  
  p <- ggplot(dat %>%
                filter(Monitor == 2)%>%
                group_by(buffer, objecive, Bhat)%>%
                summarize(CV = min(CV)),
              aes(x = Bhat, y = CV,
                  color = objecive))+
    geom_line()+
    facet_wrap(~buffer)+
    scale_color_manual(values = PNWColors::pnw_palette("Bay", n=3),
                       name = "")+
    theme_classic()+
    xlab(TeX("Expected biomass: $\\hat{B}_t$"))+
    ylab(TeX("Uncertainty: $CV_t$"))+
    ylim(0,1)+
    xlim(0,100)+
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 18),
          legend.text = element_text(size = 16),
          legend.title = element_text(size = 2),
          strip.text = element_text(size = 18),
          legend.position = "bottom",
          legend.direction = "vertical")

  return(p)
}



file1<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.4_Bthreshold_0.0_NCV_0.0_c2_0.0.csv"
file2<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.5_Bthreshold_0.0_NCV_0.0_c2_0.0.csv"
file3<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.4_Bthreshold_0.0_NCV_0.0_c2_0.05.csv"
file4<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.5_Bthreshold_0.0_NCV_0.0_c2_0.05.csv"
file5<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.4_Bthreshold_0.0_NCV_10_c2_0.0.csv"
file6<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.5_Bthreshold_0.0_NCV_10_c2_0.0.csv"
plot_thresholds(file1,file2,file3,file4,file5,file6)



file1<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.4_Bthreshold_1.0_NCV_0.0_c2_0.0.csv"
file2<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.5_Bthreshold_1.0_NCV_0.0_c2_0.0.csv"
file3<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.4_Bthreshold_1.0_NCV_0.0_c2_0.05.csv"
file4<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.5_Bthreshold_1.0_NCV_0.0_c2_0.05.csv"
file5<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.4_Bthreshold_1.0_NCV_10_c2_0.0.csv"
file6<-"~/github/KalmanFilterPOMDPs/FARM/data/HCR_policy_pstar_0.5_Bthreshold_1.0_NCV_10_c2_0.0.csv"
plot_thresholds(file1,file2,file3,file4,file5,file6)
