#######################################################################
# SMM943 Financial Econometrics - Coursework                          #
# Author: Emilio Luis Saenz Guillen, PhD Student in Actuarial Science #
# Professor: Giovanni Urga                                            #
# Bayes Business School                                               #  
# December 2022                                                       #
#######################################################################
pacman::p_load(readxl,dplyr)
rm(list = ls())

setwd("C:\\Users\\MEF08\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Simulation")
input <- "C:\\Users\\MEF08\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Simulation\\input\\glcrealddata"
output <- "C:\\Users\\MEF08\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Simulation\\output"


data_1 <- read_xlsx(file.path(input,paste0("GLC Real daily data_1985 to 1989.xlsx")),
                    sheet = "2.  real fwd curve",
                    range="A6:AU1309")
names(data_1)[names(data_1) == "31048"] <- "date"
  
data_2 <- read_xlsx(file.path(input,paste0("GLC Real daily data_1990 to 1994.xlsx")),
                    sheet = "2.  real fwd curve",
                    range="A6:AU1310")
names(data_2)[names(data_2) == "32874"] <- "date"

data_3 <- read_xlsx(file.path(input,paste0("GLC Real daily data_1995 to 1999.xlsx")),
                    sheet = "2.  real fwd curve",
                    range="A6:AU1310")
names(data_3)[names(data_3) == "34701"] <- "date"

data_4 <- read_xlsx(file.path(input,paste0("GLC Real daily data_2000 to 2004.xlsx")),
                    sheet = "2.  real fwd curve",
                    range="A6:AU1310")
names(data_4)[names(data_4) == "36528"] <- "date"

data_5 <- read_xlsx(file.path(input,paste0("GLC Real daily data_2005 to 2015.xlsx")),
                    sheet = "2. fwd curve",
                    range="A6:AU2874")
names(data_5)[names(data_5) == "38355"] <- "date"

data_6 <- read_xlsx(file.path(input,paste0("GLC Real daily data_2016 to present.xlsx")),
                    sheet = "2. fwd curve",
                    range="A6:BY1809")
names(data_6)[names(data_6) == "42370"] <- "date"

data_6 <- data_6[ , -((ncol(data_6) - 29):ncol(data_6))]


yields <- rbind(data_1,data_2,data_3,data_4,data_5,data_6)

source("chartSeries3d.alpha.R")
library(xts)
TR <- getUSTreasuries("2008")
chartSeries3d0(TR)