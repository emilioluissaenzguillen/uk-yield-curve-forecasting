#######################################################################
# SMM943 Financial Econometrics - Coursework                          #
# Author: Emilio Luis Saenz Guillen, PhD Student in Actuarial Science #
# Professor: Giovanni Urga                                            #
# Bayes Business School                                               #
# December 2022                                                       #
#######################################################################

options(repos = c(CRAN = "https://cloud.r-project.org"))

pacman::p_load(readxl, dplyr, ggplot2, ggpubr, tidyr)
rm(list = ls())

setwd("C:\\Users\\emili\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Empirical application")
input <- "C:\\Users\\emili\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Empirical application\\input\\glcrealddata"
output <- "C:\\Users\\emili\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Empirical application\\output"

## Diebold-Li

# Get the testing dates first
yields <- read.csv(file.path(output, "yields.csv"))
yields$Date <- as.Date(yields$Date)
yields <- yields[!(yields$Date < as.Date("2021-11-25")), ]

model <- read.csv(file.path(output, "MMSEForecasts.csv"), header = FALSE)
colnames(model) <- c(
  "X48_model","X54_model","X60_model","X66_model","X72_model","X78_model",
  "X84_model","X90_model","X96_model","X102_model","X108_model","X114_model",
  "X120_model","X126_model","X132_model","X138_model","X144_model","X150_model",
  "X156_model","X162_model","X168_model","X174_model","X180_model","X186_model",
  "X192_model","X198_model","X204_model","X210_model","X216_model","X222_model",
  "X228_model","X234_model","X240_model","X246_model","X252_model","X258_model",
  "X264_model","X270_model","X276_model","X282_model","X288_model","X294_model",
  "X300_model"
)

# 4 years
df_4 <- data.frame(
  Date = yields$Date,
  `Actual forward curve` = yields$X48,
  `Diebold Li SSM` = model$X48_model
)

df_4 <- df_4 %>%
  gather(key = "variable", value = "value", -Date) %>%
  arrange(Date)

a <- ggplot(df_4, aes(x = Date, y = value, group = variable)) +
  geom_line(aes(linetype = variable, color = variable)) +
  xlab("Date") +
  ylab("Real forward curve") +
  ggtitle("4 years") +
  scale_color_manual(values = c("steelblue", "firebrick1")) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# 15 years
df_15 <- data.frame(
  Date = yields$Date,
  `Actual forward curve` = yields$X180,
  `Diebold Li SSM` = model$X180_model
)

df_15 <- df_15 %>%
  gather(key = "variable", value = "value", -Date) %>%
  arrange(Date)

b <- ggplot(df_15, aes(x = Date, y = value, group = variable)) +
  geom_line(aes(linetype = variable, color = variable)) +
  xlab("Date") +
  ylab("Real forward curve") +
  ggtitle("15 years") +
  scale_color_manual(values = c("steelblue", "darkseagreen")) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  theme_minimal() +
  theme(legend.title = element_blank())

# 30 years
df_30 <- data.frame(
  Date = yields$Date,
  `Actual forward curve` = yields$X300,
  `Diebold Li SSM` = model$X300_model
)

df_30 <- df_30 %>%
  gather(key = "variable", value = "value", -Date) %>%
  arrange(Date)

c <- ggplot(df_30, aes(x = Date, y = value, group = variable)) +
  geom_line(aes(linetype = variable, color = variable)) +
  xlab("Date") +
  ylab("Real forward curve") +
  ggtitle("30 years") +
  scale_color_manual(values = c("steelblue", "mediumorchid1")) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  theme_minimal() +
  theme(legend.title = element_blank())

ggarrange(a, b, c, ncol = 3, nrow = 1, common.legend = TRUE, legend = "bottom")


## Diebold-Li: equivalent rate
data <- read.csv(file.path(output,paste0("equiv_rate_merged_dbli.csv")))
names(data)[names(data) == "X"] <- "Date"
data$Date <- as.Date(data$Date)

names(data)[names(data) == "equiv_rate"] <- "Actual equivalent rate"
names(data)[names(data) == "equiv_rate_model_dbli"] <- "Testing equivalent rate"

df <- data %>%
  gather(key = "variable", value = "value", -Date)%>%
  arrange(Date)


ggplot(df, aes(x=Date, y=value, group=variable)) +
  geom_line(aes(linetype=variable, color=variable))+
  xlab('Date') +
  ylab('Real equivalent rate') + 
  scale_color_manual(values = c("steelblue","darkorange"))+
  scale_linetype_manual(values=c("solid", "dashed")) +
  theme_minimal() +
  theme(legend.title=element_blank())


## LSTM
data <- read.csv(file.path(output,paste0("merged.csv")))
names(data)[names(data) == "X"] <- "Date"
data$Date <- as.Date(data$Date)

# 4 years maturity
df <- data[,c("Date","X48_actual","X48_train","X48_test")]
names(df)[names(df) == "X48_actual"] <- "Actual forward curve"
names(df)[names(df) == "X48_train"] <- "LSTM (training data)"
names(df)[names(df) == "X48_test"] <- "LSTM (testing data)"

df <- df %>%
  gather(key = "variable", value = "value", -Date)%>%
  arrange(Date) %>%
  filter(!is.na(value))

ggplot(df, aes(x=Date, y=value, group=variable)) +
  geom_line(aes(linetype=variable, color=variable))+
  xlab('Date') +
  ylab('Real forward curve') + 
  scale_color_manual(values = c("steelblue","firebrick1","darkseagreen"))+
  scale_linetype_manual(values=c("solid",  "dashed","solid")) +
  theme_minimal() +
  theme(legend.title=element_blank())

# 15 years maturity
df <- data[,c("Date","X180_actual","X180_train","X180_test")]
names(df)[names(df) == "X180_actual"] <- "Actual forward curve"
names(df)[names(df) == "X180_train"] <- "LSTM (training data)"
names(df)[names(df) == "X180_test"] <- "LSTM (testing data)"

df <- df %>%
  gather(key = "variable", value = "value", -Date)%>%
  arrange(Date) %>%
  filter(!is.na(value))

ggplot(df, aes(x=Date, y=value, group=variable)) +
  geom_line(aes(linetype=variable, color=variable))+
  xlab('Date') +
  ylab('Real forward curve') + 
  scale_color_manual(values = c("steelblue","gold","darkseagreen"))+
  scale_linetype_manual(values=c("solid",  "dashed","solid")) +
  theme_minimal() +
  theme(legend.title=element_blank())

# 25 years maturity
df <- data[,c("Date","X300_actual","X300_train","X300_test")]
names(df)[names(df) == "X300_actual"] <- "Actual forward curve"
names(df)[names(df) == "X300_train"] <- "LSTM (training data)"
names(df)[names(df) == "X300_test"] <- "LSTM (testing data)"

df <- df %>%
  gather(key = "variable", value = "value", -Date)%>%
  arrange(Date) %>%
  filter(!is.na(value))

ggplot(df, aes(x=Date, y=value, group=variable)) +
  geom_line(aes(linetype=variable, color=variable))+
  xlab('Date') +
  ylab('Real forward curve') + 
  scale_color_manual(values = c("steelblue","mediumorchid1","darkseagreen"))+
  scale_linetype_manual(values=c("solid",  "dashed","solid")) +
  theme_minimal() +
  theme(legend.title=element_blank())

  
## LSTM: equivalent rate
data <- read.csv(file.path(output,paste0("equiv_rate_merged.csv")))
names(data)[names(data) == "X"] <- "Date"
data$Date <- as.Date(data$Date)

df <- data
df <- df %>% 
  mutate(training = if_else(Date<as.Date("2021-11-25"),equiv_rate_model,0))

df["training"][df["training"] == 0] <- NA
df <- df %>% 
  mutate(testing = if_else(Date>=as.Date("2021-11-25"),equiv_rate_model,0))

df["testing"][df["testing"] == 0] <- NA


df <- df[,c("Date","equiv_rate","training","testing")]
names(df)[names(df) == "equiv_rate"] <- "Actual equivalent rate"
names(df)[names(df) == "training"] <- "Training equivalent rate"
names(df)[names(df) == "testing"] <- "Testing equivalent rate"

df <- df %>%
  gather(key = "variable", value = "value", -Date)%>%
  arrange(Date) %>%
  filter(!is.na(value))


ggplot(df, aes(x=Date, y=value, group=variable)) +
  geom_line(aes(linetype=variable, color=variable))+
  xlab('Date') +
  ylab('Real equivalent rate') + 
  scale_color_manual(values = c("steelblue","darkorange","darkseagreen"))+
  scale_linetype_manual(values=c("solid",  "dashed","solid")) +
  theme_minimal() +
  theme(legend.title=element_blank())

