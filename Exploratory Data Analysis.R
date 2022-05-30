curdir <- getwd()
setwd(curdir)

# import original data set
UberLyft <- read.csv("Uber and Lyft Rideshare - Boston - 2018.csv")

# insert a new variable Surge Multiplier 
UberLyft$NewSurgeMult <- as.factor(ifelse(UberLyft$surge_multiplier > 1, 1, 0))

# import main data set
cab <- read.csv("LyftFinal.csv")

# import libraries
library(ggplot2)
library(tidyverse)
library(tinytex)
library(reshape2)


######## OVERVIEW - DATA IMBALANCE ############################################

# prepare the data frame for the bar chart
bar_overview <- UberLyft %>%
  group_by(cab_type, NewSurgeMult) %>%
  summarise(count = n()) %>% # compute total records per each bar
  mutate(NewSurgeMult=factor(NewSurgeMult, levels = c(1,0), ordered = T)) # reorder the levels of "NewSurgeMult"

# create the stacked bar chart
barchart_overview <- bar_overview %>%
  ggplot(aes(y=count, x=cab_type)) +
  geom_col(aes(fill=NewSurgeMult), width=0.7, alpha=0.7) +
  labs(x = "Ride Count",
       y = "",
       title = "RIDE COUNT BY CAB TYPE AND BY SURGE MULTIPLIER") +
  theme_minimal() +
  theme(axis.title.x = element_text(margin = margin(t = 10)), # x-axis title is too close to axis ticks labels,
        axis.text = element_text(size=10),
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.x = element_blank(), # change color of grid lines
        panel.grid.major.y = element_blank()) + # remove horizontal grid lines
  scale_fill_manual("Surge Multiple", values = c("#AA336A", "grey"), labels=c("Surged", "Non-surge")) 
barchart_overview


######## OVERVIEW - CORRELATION HEATMAP ############################################
# create a list of numerical variables
num_col <- unlist(lapply(cab,is.numeric)) # creatE a list of names of columns that have numeric values
cab_num <- cab[,num_col]

cormat <- cor(cab_num)
head(cormat)
melted_cormat <- melt(cormat)
head(melted_cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
upper_tri
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)

ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
heatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "dark green", high = "darkred", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1))+
  labs(title="CORRELATIONS AMONG VARIABLES") +
  theme(
    plot.title = element_text(size=13),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))
heatmap

######### OVERVIEW - RIDE COUNT OVER THE TIME FRAME ###########################################
# prepare the data frame for the bar chart
bar_date <- cab %>%
  group_by(month, day) %>%
  summarise(count = n()) %>% # compute total records per each bar
  unite("fullDate", month, day, sep = "/")%>%
  mutate(date = format(as.Date(fullDate, format='%m/%d'),'%m/%d'))

(bar_date_mean <- mean(bar_date$count))

# create the bar chart
barchart_date <- bar_date %>%
  ggplot(aes(x=date, y=count)) +
  geom_col(fill = ifelse(bar_date$count>13684, "#AA336A", "grey"), alpha=0.7) + 
  theme_grey(base_size = 14) +
  labs(x = "date",
       y = "Ride Count",
       title = "RIDE COUNT OVER THE TIME FRAME") +
  theme_minimal() +
  theme(axis.title.x = element_text(margin = margin(t = 10)), # x-axis title is too close to axis ticks labels
        axis.title.y = element_text(margin = margin(t = 10)), # y-axis title is too close to axis ticks labels
        axis.text = element_text(size=10),
        axis.text.x = element_text(angle=45),
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.x = element_blank(), # remove major grid lines
        panel.grid.major.y = element_blank()) + # remove major grid lines
  geom_hline(yintercept = bar_date_mean, color="black", linetype = 'dotted') + # add an average line
  annotate("text", x = 10.5, y = 14800, label = "Average Ride Count\nper Day = 13,684", vjust = -0.5) # add annotation
barchart_date

######### PRICE DISTRIBUTION PER CAB TYPE ###########################################
boxplot <- cab %>%
  ggplot(aes(x=price, y=full_cab_name))+
  geom_jitter(width = 0.05, height = 0, color="grey80") +
  geom_boxplot(fill="#AA336A", alpha = 0.5, outlier.color = NA) +
  labs(title="PRICE DISTRIBUTION BY CAB TYPE") + # add chart title
  theme(axis.ticks = element_blank(), # remove ticks
        axis.title.y = element_blank(), # remove y axis title
        plot.caption = element_text(face="italic", margin = margin(t = 10)), # plot caption to close to x-axis
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.y = element_line(color="grey90"),
        panel.grid.major.x = element_blank()) # remove vertical major grid lines
boxplot

######### RIDE COUNT BY CAB TYPE ###########################################
# prepare the data frame for the bar chart
bar_cabType <- cab %>%
  group_by(full_cab_name, NewSurgeMult) %>%
  summarise(count = n()) %>% # compute total records per each bar
  mutate(NewSurgeMult=factor(NewSurgeMult, levels = c(1,0), ordered = T)) # reorder the levels of "NewSurgeMult"

# create the stacked bar chart
barchart_cabType <- bar_cabType %>%
  ggplot(aes(x=count, y=full_cab_name)) +
  geom_col(aes(fill=NewSurgeMult), width=0.7, alpha=0.7) +
  labs(x = "Ride Count",
       y = "",
       title = "RIDE COUNT BY CAB TYPE AND BY SURGE MULTIPLE") +
  theme_minimal() +
  theme(axis.title.x = element_text(margin = margin(t = 10)), # x-axis title is too close to axis ticks labels,
        axis.text = element_text(size=10),
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.x = element_blank(), # change color of grid lines
        panel.grid.major.y = element_blank()) + # remove horizontal grid lines
  scale_fill_manual("Surge Multiple", values = c("#AA336A", "grey"), labels=c("Surged", "Non-surge")) 
barchart_cabType


###### AVERAGE SURGED  RIDE COUNT BY HOUR OF THE DAY #################
# count the number of distinct days over the time period
dayCount <- cab %>%
  filter(surge_multiplier==1) %>%
  group_by(month,day) %>%
  summarise(count = n())
dayCount <- nrow(dayCount)

# prepare the data frame for the bar chart
bar_hour_a <- cab %>%
  filter(surge_multiplier==1) %>%
  group_by(hour) %>%
  summarise(count = n()/dayCount) # compute total records per each bar

#calculate mean of count
(hour_a_mean <- mean(bar_hour_a$count))

# create the bar chart
barchart_hour_a <- bar_hour_a %>%
  ggplot(aes(x=hour, y=count)) +
  geom_col(fill = ifelse(bar_hour_a$count>529, "#AA336A", "grey"), alpha=0.7) + 
  labs(x = "Hour of the Day",
       y = "Ride Count",
       title = "AVERAGE SURGED RIDE COUNT BY HOUR OF THE DAY") +
  theme_minimal() +
  theme(axis.title.x = element_text(margin = margin(t = 10)), # x-axis title is too close to axis ticks labels
        axis.title.y = element_text(margin = margin(t = 10)), # y-axis title is too close to axis ticks labels
        axis.text = element_text(size=10),
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.x = element_blank(), # remove major grid lines
        panel.grid.major.y = element_blank()) + # remove major grid lines
  scale_x_continuous(breaks = seq(0,23,1)) + # define the values of x sticks
  geom_hline(yintercept = hour_a_mean, color="black", linetype = 'dotted') + # add an average line
  annotate("text", x = 6, y = 570, label = "Average Surged Ride Count = 529", vjust = -0.5) # add annotation
barchart_hour_a


###### AVERAGE SURGED  RIDE COUNT BY DAY OF THE WEEK #################
# count the total number of rides per weekday
bar_day_a1 <- cab %>%
  filter(NewSurgeMult==1) %>%
  select(weekday,day) %>%
  group_by(weekday) %>%
  summarize(count = n())

# count the frequency of a weekday
bar_day_a2 <- cab %>%
  select(weekday, day) %>%
  group_by(weekday) %>%
  dplyr::mutate(count = n_distinct(day)) %>% # compute total records per each bar
  select(-day) %>% # drop the day column
  distinct(weekday, count)
bar_day_a2

# combine two data frames
bar_day_a3 <- merge(bar_day_a1, bar_day_a2, by="weekday")
bar_day_a3$avgCount <- (bar_day_a3$count.x)/(bar_day_a3$count.y)
(bar_day_a3_mean <- mean(bar_day_a3$avgCount))
bar_day_a4 <- bar_day_a3 %>%
  mutate(weekday=factor(weekday, levels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"), ordered = T))


# create the bar chart
barchart_day <- bar_day_a4 %>%
  ggplot(aes(x=weekday, y=avgCount)) +
  geom_col(fill = ifelse(bar_day_a4$avgCount>1097, "#AA336A", "grey"), alpha=0.7) + 
  labs(x = "",
       y = "Ride Count",
       title = "AVERAGE SURGED RIDE COUNT BY WEEKDAY") +
  theme_minimal() +
  theme(axis.title.x = element_text(margin = margin(t = 10)), # x-axis title is too close to axis ticks labels
        axis.title.y = element_text(margin = margin(t = 10)), # y-axis title is too close to axis ticks labels
        axis.text = element_text(size=10),
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.x = element_blank(), # remove major grid lines
        panel.grid.major.y = element_blank()) + # remove major grid lines
  geom_hline(yintercept = bar_day_a3_mean, color="black", linetype = 'dotted') + # add an average line
  annotate("text", x = 6, y = 1100, label = "Average Surged Ride Count = 1,097", vjust = -0.5) # add annotation
barchart_day

###### AVERAGE SURGED  RIDE COUNT BY WEATHER CONDITION #################
# count total frequency
(bar_weather_a1 <- cab %>%
   group_by(short_summary) %>%
   summarise(count = n())) # compute total records per each bar

# calculate mean of count
(hour_weather_mean <- mean(bar_weather_a1$count))

bar_weather_a2 <- cab %>%
  filter(NewSurgeMult==1) %>%
  select(short_summary,day) %>%
  group_by(short_summary) %>%
  dplyr::mutate(count = n_distinct(day)) %>% # compute total records per each bar
  select(-day) %>% # drop the day column
  distinct(short_summary, count)
bar_weather_a2

# combine two data frames
bar_weather_a3 <- merge(bar_weather_a1, bar_weather_a2, by="short_summary")
bar_weather_a3$avgCount <- (bar_weather_a3$count.x)/(bar_weather_a3$count.y)
(bar_weather_a3_mean <- mean(bar_weather_a3$avgCount))

# create the bar chart
barchart_weather_a <- bar_weather_a3 %>%
  ggplot(aes(x=short_summary, y=avgCount)) +
  geom_col(fill = ifelse(bar_weather_a3$avgCount>2902, "#AA336A", "grey"), alpha=0.7) + 
  labs(x = "Weather Condition",
       y = "Ride Count",
       title = "AVERAGE SURGED RIDE COUNT BY WEATHER CONDITION") +
  theme_minimal() +
  theme(axis.title.x = element_text(margin = margin(t = 10)), # x-axis title is too close to axis ticks labels
        axis.title.y = element_text(margin = margin(t = 10)), # y-axis title is too close to axis ticks labels
        axis.text = element_text(size=10),
        plot.title = element_text(size=14, hjust=0.5),
        panel.grid.minor = element_blank(), # remove minor grid lines
        panel.grid.major.x = element_blank(), # remove major grid lines
        panel.grid.major.y = element_blank()) + # remove major grid lines
  geom_hline(yintercept = bar_weather_a3_mean, color="black", linetype = 'dotted') + # add an average line
  annotate("text", x = 2, y = 3000, label = "Average Surged Ride Count = 2,902", vjust = -0.5) # add annotation
barchart_weather_a
