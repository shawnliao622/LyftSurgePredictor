# Read data
Cab <- read.csv("Uber and Lyft Rideshare - Boston - 2018.csv")

# Remove all the unneccessary columns
Cab$id <- NULL
Cab$timestamp <- NULL
Cab$timezone <- NULL
Cab$product_id <- NULL
Cab$long_summary <- NULL
Cab$windGustTime <- NULL
Cab$temperatureHighTime <- NULL
Cab$temperatureLowTime <- NULL
Cab$apparentTemperatureHighTime <- NULL
Cab$apparentTemperatureLowTime <- NULL
Cab$icon <- NULL
Cab$visibility <- NULL
Cab$visibility.1 <- NULL
Cab$sunriseTime <- NULL
Cab$sunsetTime <- NULL
Cab$moonPhase <- NULL
Cab$uvindexTime <- NULL
Cab$temperatureMinTime <- NULL
Cab$temperatureMaxTime <- NULL
Cab$apparentTemperatureMinTime <- NULL
Cab$apparentTemperatureMaxTime <- NULL
Cab$apparentTemperature<- NULL
Cab$precipIntensity<- NULL
Cab$humidity<- NULL
Cab$windGust<- NULL
Cab$temperatureHigh<- NULL
Cab$temperatureLow<- NULL
Cab$apparentTemperatureHigh<- NULL
Cab$apparentTemperatureLow<- NULL
Cab$dewPoint<- NULL
Cab$pressure<- NULL
Cab$windBearing<- NULL
Cab$cloudCover<- NULL
Cab$uvIndex<- NULL
Cab$ozone<- NULL
Cab$moonPhase<- NULL
Cab$precipIntensityMax<- NULL
Cab$temperatureMin<- NULL
Cab$temperatureMax<- NULL
Cab$apparentTemperatureMin<- NULL
Cab$apparentTemperatureMax<- NULL
Cab$uvIndexTime<- NULL
Cab$windSpeed<- NULL

# Remove taxi data
Cab <- Cab[which(Cab$name != "Taxi"),]

# Remove Uber data as it only has one class of surge multiple
Cab <- Cab[which(Cab$cab_type != "Uber"),]

#install.packages("lubridate")
library(lubridate)

Cab$datetime <- as.POSIXct(Cab$datetime,
                           tz = Sys.timezone())

# Create a new variable to label the day of week
Cab$weekday <- wday(Cab$datetime, label=TRUE)

breaks <- hour(hm("00:00", "6:00", "12:00", "18:00", "23:59"))
# Labels for the breaks
labels <- c("Night", "Morning", "Afternoon", "Evening")

# Create a new variable to label time of the day (Night, Morning, Afternoon, Evening) 
Cab$time_of_day <- cut(x=Cab$hour, breaks = breaks, labels = labels, include.lowest=TRUE)

# Create a binary variable to see if it's in rush hour 
# Rush hour is 6am-10am and 3pm-7pm according to Boston Region MPO
Cab$rush_hour <- ifelse(Cab$hour >= 6 & Cab$hour < 10|Cab$hour >= 15 & Cab$hour < 19,
                        1, 0)
Cab$Weekday_Dummy <- ifelse(Cab$weekday=="Sat" | Cab$weekday=="Sun", 1, 0) 

# Create a binary variable to see if it's a surge
Cab$NewSurgeMult <- ifelse(Cab$surge_multiplier > 1, 1, 0)
price <- read.csv("Pricing Policy.csv") 

# show names of columns in the "price" data set
colnames(price)

# remove "Lyft" in column "name" in the main data set
Cab$name <- gsub(pattern = "Lyft", replacement="", Cab$name , perl=T )

# create a new column named "full_cab_name" in the main data set whose values are the combination of column "cab_type" and column "name" in the main data set.
Cab$full_cab_name <- paste(Cab$cab_type, Cab$name)

## check one more time to see if values in column "full-cab_name" in the main data set are the same as values in column "full-cab_name" in the "price" data set.
# show unique values in column "full_cab_name" in the main data set
unique(Cab$full_cab_name)
# show unique values in column "full_cab_name" in the "price" data set
unique(price$full_cab_name)

# assign corresponding values from the "price" data set to the main data set by looking up the values in column "full_cab_name"
df_Cab_Final <- merge(Cab,price,by="full_cab_name")
df_Cab_Final$name <- NULL

df_Model <- df_Cab_Final
df_Model$datetime <- NULL
df_Model$hour <- NULL
df_Model$day <- NULL
df_Model$weekday <- NULL
df_Model$surge_multiplier <- NULL
df_Model$latitude <- NULL
df_Model$longitude <- NULL
df_Model$base_fare  <- NULL       
df_Model$cost_per_mile  <- NULL  
df_Model$cost_per_minute  <- NULL
df_Model$minimum_fare <- NULL  
df_Model$service_fee <- NULL
df_Model$price <- NULL

str(df_Model)

for (name in colnames(df_Model[, -which(colnames(df_Model) =="NewSurgeMult")])) {
  if (class(df_Model[, name])=="character" | class(df_Model[, name])=="integer"){
    df_Model[, name] <- as.factor(df_Model[, name])
  }
}

# 256173 data for Lyft
df_Lyft <- df_Model[df_Model$cab_type == "Lyft", ]
df_Lyft$cab_type <- NULL
nrow(df_Lyft)

# Partition Lyft and Uber data 
set.seed(758)

#install.packages("caret")
library(caret)

# Reduce the data size to 1 / 10
randomSample <- createDataPartition(df_Lyft$NewSurgeMult, p = 0.1, list = F)

df_Lyft <- df_Lyft[randomSample, ]
inTrain <- createDataPartition(df_Lyft$NewSurgeMult, p = 0.7, list = F)
df_Train_Lyft <-df_Lyft[inTrain, ]
df_Valid_Lyft <-df_Lyft[-inTrain, ]

# Oversampling to deal with imbalanced data (only for training data)
# https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/

#install.packages("ROSE")
library(ROSE)
df_Over_train <- ovun.sample(NewSurgeMult~., method="over", data=df_Train_Lyft)
df_Over_train <- as.data.frame(df_Over_train$data)

# Check if the training data is balanced
summary(df_Over_train)
summary(df_Valid_Lyft)

library("writexl")
write_xlsx(df_Over_train, "df_Over_train.xlsx")
write_xlsx(df_Valid_Lyft, "df_Valid_Lyft.xlsx")
