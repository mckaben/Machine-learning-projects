# Philippe Kabenla
# DSC-607
#final project
# Module 8

library(readr)
cardio <- read_csv("C:/Users/philippe/Documents/data science 3/cardio.csv")
View(cardio)
head(cardio)
str(cardio)
sum(is.na(cardio))

#let's extract few variables
library(tidyverse)
cardio2 <- select(cardio, 1,2,5,7,12)
view(cardio2)
summary(cardio)
cardio2$cardio[cardio2$cardio == 1] <- "yes"
cardio2$cardio[cardio2$cardio == 0] <- "no"
with(cardio2,hist(age))
with(cardio2,hist(cholesterol), fill = cardio)
with(cardio2,hist(gender), fill = cardio)
with(cardio2,hist(ap_hi), fill = cardio)
#scatter plots of the DV with the IV(s)
library(ggpubr)
#let's plot cardio and ap_hi. 
ggplot(cardio, aes(x = ap_hi, y = cardio)) +
  geom_point() +
  stat_smooth()

#let's plot age and cardio
ggplot(cardio, aes(x = age, y = cardio)) +
  geom_point() +
  stat_smooth()

#let's plot weight and cardio
ggplot(cardio, aes(x = weight, y = cardio)) +
  geom_point() +
  stat_smooth()

#Summary statistics
library(reshape)
library(reshape2)
library(ggplot2)
library(GGally)
library(psych)
#some correlation plots and histograms
pairs.panels(cardio2[1:5])

# Nice visualization of correlations
ggcorr(cardio2, method = c("everything", "pearson"))

ggplot(cardio2, aes(x=ap_hi, fill=cardio)) + 
  geom_bar() +
  xlab("cardiovascular disease") +
  ylab("Count") +
  ggtitle("Analysis of Presence and Absence of cardiovascular") +
  scale_fill_discrete(name = "cardiovascular")

#Distribution of Male and Female population across Age parameter
ggplot(cardio2,aes(x=cholesterol,fill=cardio))+
geom_histogram()+
xlab("cholesterol") + 
ylab("Number")+
guides(fill = guide_legend(title = "cardio"))

library(gridExtra)

grid.arrange(
  ggplot(cardio2, aes(x = gender, fill = cardio))+
    geom_bar(position = "fill"),
  
  ggplot(cardio2, aes(x = cholesterol, fill = cardio))+
    geom_bar(position = "fill"),
  
  ggplot(cardio2, aes(x = ap_hi, fill = cardio))+
    geom_bar(position = "fill"), nrow = 3)

cardio2$cardio <- ifelse(cardio2$cardio == "no",0,1)


pairs.panels(cardio2[1:5])

# Nice visualization of correlations
ggcorr(cardio2, method = c("everything", "pearson"))
boxplot(cardio2)
#summary of the data

#Age | Objective Feature | age | int (days)
#Height | Objective Feature | height | int (cm) |
# Weight | Objective Feature | weight | float (kg) |
#Gender | Objective Feature | gender | categorical code |
#Systolic blood pressure | Examination Feature | ap_hi | int |
#Diastolic blood pressure | Examination Feature | ap_lo | int |
#Cholesterol | Examination Feature | cholesterol | 1: normal,
#2: above normal, 3: well above normal |
#Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 
#3: well above normal |
#Smoking | Subjective Feature | smoke | binary |
#Alcohol intake | Subjective Feature | alco | binary |
#Physical activity | Subjective Feature | active | binary |
#Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

# let's check for missing data
sum(is.na(cardio2))

#let's normalize the data
normalize <- function(x) {
  ((x -min(x)) / (max(x) - (min(x))))
}

cardio <- as.data.frame(cardio)
cardio2 <- as.data.frame(cardio2)
#let's compute a logistic regression
mod1 <- glm(cardio ~., data = cardio, 
            family = binomial(link = "logit"))
mod1
summary(mod1)

stepwise <- step(mod1)
best_mod <- glm(cardio ~ age + weight + ap_hi + ap_lo + cholesterol +
                  active, data = cardio, family = binomial(link = "logit"))
best_mod  
summary(best_mod)

#plot the graph of age vs cardio
ggplot(cardio, aes(age, cardio)) +
  geom_point(alpha = 0.8) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Logistic Regression Model", 
    x = "predictor age",
    y = "Probability of having cardiovascular disease "
  )

# plot the graph of cholesterol vs cardio

ggplot(cardio, aes(cholesterol, cardio)) +
  geom_point(alpha = 0.8) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Logistic Regression Model", 
    x = "predictor cholesterol",
    y = "Probability of having cardiovascular disease "
  )

#plot the graph of ap_hi vs cardio
ggplot(cardio, aes(ap_hi, cardio)) +
  geom_point(alpha = 0.8) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Logistic Regression Model", 
    x = "predictor ap_hi",
    y = "Probability of having cardiovascular disease "
  )
#The following shows the diagnostic plots of the model.
plot_numbers <- 1:6
layout(matrix(plot_numbers, ncol = 2, byrow = TRUE))
plot(best_mod, plot_numbers)

#information about the data

library(caret)
library(tidyverse)

set.seed(123)
### lets train the data and create the index matrix



index <- createDataPartition(cardio$cardio, p=.8,
                             list = FALSE, times = 1)


train_cardio <- cardio[index,] # 80% data training
test_cardio <- cardio[-index,] # 20% remaining test data

train_cardio_labels <- cardio[index,12] 
test_cardio_labels <- cardio[-index,12]
# Load class package
library(class)

##################   KNN CLASSIFIER   ####################
#Creating separate dataframe for 'Credibility' feature which is our target.
#Find the number of observation
NROW(train_cardio_labels)
# the number of observation is 3377.
#So, we have 3377 observations in our training data set. The square root of 3377 is 
#around 58.11, therefore we will create two models. One with 'K' value as 58 and 
#the other model with a 'K' value as 59.


knn.58 <- knn(train=train_cardio, test=test_cardio, cl=train_cardio_labels, k=58)
knn.59 <- knn(train=train_cardio, test=test_cardio, cl=train_cardio_labels, k=59)
#After building the model, it is time to calculate the accuracy of the created models.
#Calculate the proportion of correct classification for k = 236, 237

ACC.58 <- 100 * sum(test_cardio_labels == knn.58)/NROW(test_cardio_labels)
ACC.59 <- 100 * sum(test_cardio_labels == knn.59)/NROW(test_cardio_labels)

ACC.58              
ACC.59

#As shown above, the accuracy for K = 58 is 63.63 and for K = 59 it is 63.15.

# let's check the predicted outcome against the actual value in tabular form.
#prediction against actual value in tabular form for k=236
table(knn.58 ,test_cardio_labels)
# 0 = no, and 1 = yes
#as you can see 284 out of 844 were accurately predicted true negative (TN), meaning that
#284 cases were predicted to be negative of cardiovascular disease. Also,
#253 out of 844 were accurately predicted true positive (TP), meaning that 253 cases
# were predicted to be having cardiovascular disease. 133 cases were false negative 
#and 174, false positive.

#prediction against actual value in tabular form for k=58
table(knn.59 ,test_cardio_labels)

#Knn model
knn.mod <- knn(train=train_cardio,
               test=test_cardio, cl=train_cardio_labels, k=58)
k.optm <- 100 * sum(test_cardio_labels == knn.mod)/NROW(test_cardio_labels)
#######Optimization
#which 'K' value will result in the most accurate model.
i=1
k.optm=1
for (i in 1:60){
  knn.mod <- knn(train=train_cardio, test=test_cardio, cl=train_cardio_labels, k=i)
  k.optm[i] <- 100 * sum(test_cardio_labels == knn.mod)/NROW(test_cardio_labels)
  k=i
  cat(k,'=',k.optm[i],'\n')
}  

#K = 8 has the highest accuracy of 66.59
# Accuracy plot
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

#let's evaluate the model performance
library(gmodels)

#let's use k = 8
knn.pred <- knn(train=train_cardio,
                test=test_cardio, cl=train_cardio_labels, k=8)
CrossTable(x = test_cardio_labels, y = knn.pred, prop.chisq = FALSE )

###################       INTERPRETATION       ##################

#The test data consisted of 844 observations. Out of which 294 cases have been 
#accurately predicted (TN = true negative) as " not having cardiovascular  
# disease (0) which constitutes 34.8%. Also, 250 out of 844 observations
#were accurately predicted (TP = true positive) as at having the disease (1) which 
#constitutes 29.6%. Thus a total of 250 out of 844 predictions were True positive.

#177 out of 844 cases have been accurately predicted (FN = false negative),
#as at having the disease (1) which constitutes 21.00%. But, were predicted as not having the disease (0).
#there were 123 cases of (FP = false positive), meaning that 123 cases were actually
#not having the disease (0) but predicted as they were having it(1).

########### CLASSIFICATION USING DECISION TREE ##############
#Le's load the package "rpart" & "rpart.plot"
library("rpart")
library("rpart.plot")
library(DAAG)
library(party)
library(mlbench)
### lets train the data and create the index matrix

cardio$cardio[cardio$cardio ==1] <- "yes"
cardio$cardio[cardio$cardio ==0] <- "no"

index <- createDataPartition(cardio$cardio, p=.7,
                             list = FALSE, times = 1)


train_cardio <- cardio[index,] # 70% data training
test_cardio <- cardio[-index,] # 30% remaining test data
#let's check if the randomization process is correct.
prop.table(table(train_cardio$cardio))
prop.table(table(test_cardio$cardio))
#based on this train data, 49.85% were accurately predicted to not have the disease.
# and about 50.15% to have the cardiovascular disease.

# let's build and plot. First, let's create the target.

target <- rpart(cardio ~.,  
                data = train_cardio, method = "class")
rpart.plot(target,extra = 106)
printcp(target)  #display results
plotcp(target) # visualize cross-validation results
summary(target) # detailed summary of splits and result

#Based on the train data,50% of the people have the cardiovascular disease. among
#thee, people with ap_hi < 138, 37% don't have the disease while 85% do.
#from the people aged 20000 days or 54.8 years old and less, 51% have the disease
#while 27% don't. For cholesterol < 3, 75% have the disease while 47% don't.
#confusion matrix with train_heart
library(tree)
require(tree)
library(ISLR)
tree <- rpart(cardio ~.,
              data = train_cardio,cp=0.07444)
p <- predict(tree, train_cardio, type = 'class')
# with positive class "yes"
confusionMatrix(p, as.factor(train_cardio$cardio), positive = "yes")
#based on the confusion matrix with train_heart, the accuracy obtained is 0.69.
#which is very good for 80% trained of the data.

#with positive class "no"
confusionMatrix(p, as.factor(train_cardio$cardio), positive = "no")
#the accuracy is about the same, 0.69.

#make prediction
predict_unseen <-predict(target, test_cardio, type = 'class')
predict_unseen
table_mat <- table(test_cardio$cardio, predict_unseen)
table_mat
#based on these results, 502 out of 844 cases were predicted (TN = true negative).
#387 were predicted (TP = true positive), 248 cases were false negative, and 129 
#false negative.

#let's compute the accuracy test
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))
# accuracy test is 0.70

#Naive bayes on the data
library(e1071)
library(naivebayes)
#Default Paramters
NBclassfier <- naiveBayes(cardio~., data=train_cardio)
print(NBclassfier)
#based on this train data, 49.85% were accurately predicted to not have the disease.
# and about 50.15% to have the cardiovascular disease.
#######cluster##################

library(cluster) # clustering algorithms
library(factoextra) # clustering algorithms & visualization

#convert x into a numeric format
cardio$cardio <- ifelse(cardio$cardio == "no",0, 1)
cardio <- na.omit(cardio)
cardio2_sc <- scale(cardio2)
head(as_tibble(cardio2_sc))

k2 <- kmeans(cardio2_sc, 
             center = 2,
             nstart = 25  )
k2
fviz_cluster(k2, data = cardio2_sc)

# k2
k3 <- kmeans(cardio2_sc, centers = 3, nstart = 25)
k4 <- kmeans(cardio2_sc, centers = 4, nstart = 25)
k5 <- kmeans(cardio2_sc, centers = 5, nstart = 25)

p1 <- fviz_cluster(k2, geom = "point", data = cardio2_sc)+
  ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point", data = cardio2_sc)+
  ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point", data = cardio2_sc)+
  ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point", data = cardio2_sc)+
  ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1,p2,p3,p4, nrow = 2)

#function to create a plot of the number of clusters vs. 
#the total within sum of squares

fviz_nbclust(cardio2_sc, kmeans, method = "wss")
#For this plot it appear that there is a bit of an elbow or "bend" at k = 2 clusters

#other form to find the right k
fviz_nbclust(cardio2_sc, kmeans, method = "silhouette")

#gap statistic method

set.seed(123)

gap_stat <- clusGap(cardio2_sc, FUN = kmeans, nstart = 25, K.max = 10, B = 50)

print(gap_stat, method = "firstmax")

fviz_gap_stat(gap_stat)

#Based on the determination of the best k, k = 2 is the best
set.seed(123)
best1 <- kmeans(cardio2_sc, 2, nstart = 25)
best1

library(factoextra)
set.seed(123)
fviz_cluster(best1, cardio2_sc, geom = "point",
             ellipse.type = "norm")
#the two clusters here represent 2 groups of people. based on the visual above,
#these clusters can be interpreted as, cluster 2 = people with no cardiovascular 
#disease and cluster 1 = people with the disease. 

#let's make better plots with silhouette
#First, let's compute the distance matrix
dist_data <- dist(cardio2_sc, method = "euclidean")
plot(silhouette(best1$cluster, dist = dist_data), col=2:3)

#find means of each cluster
aggregate(cardio2_sc, by=list(cluster=best1$cluster), mean)

#add cluster assigment to original data
final_data <- cbind(cardio2, cluster = best1$cluster)

#view final data
head(final_data)

