options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
data(brca)
str(brca)
str(brca$x)
dim(brca$x)
table(brca$y)
mean(brca$y == "M")
which.max(colMeans(brca$x))
which.min(colSds(brca$x))
# ----------------------------
x <- brca$x
y <- brca$y
brca_x <- as.data.frame(unlist(x))
scale <- function(x){
  (x-mean(x))/sd(x)
  (x-mean(x))/sd(x)
}
x1 <- brca_x %>%  mutate_all(scale)
sd(x1[,1])
median(x1[,1])
# by sweep function
x_centered <- sweep(x, 2, colMeans(x))
x_scaled <- sweep(x_centered, 2, colSds(x), FUN = "/")
#x_scaled$y <- as.vector(unlist(brca[["y"]]))
sd(x_scaled[,1])
# ----------------------------
d_samples <- dist(x_scaled)
dist_BtoB <- as.matrix(d_samples)[1, brca$y == "B"]
mean(dist_BtoB[2:length(dist_BtoB)])
dist_BtoM <- as.matrix(d_samples)[1, brca$y == "M"]
mean(dist_BtoM)
# ----------------------------
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)
# ----------------------------
# hirericial Clustering of Feature
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)
# -----------------------------------
pca<-prcomp(x_scaled)
summary(pca)
# --------------------------------------
data.frame(pca$x[,1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()
# -------------------------------------
data.frame(type = brca$y, pca$x[,1:10]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()
# --------------------------------------
# set.seed(1) if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]
mean(train_y=="B")
mean(test_y=="B")
# ---------------------------------------
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}
# ---------------------------------------
set.seed(3, sample.kind = "Rounding")    # if using R 3.6 or later
k <- kmeans(train_x, centers = 2)
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")
mean(kmeans_preds == test_y)
table(test_y,kmeans_preds)
sensitivity(factor(kmeans_preds), test_y, positive = "B")
sensitivity(factor(kmeans_preds), test_y, positive = "M")
# ---------------------------------------
train_glm <- train(train_x, train_y,
                   method = "glm")
glm_preds <- predict(train_glm, test_x)
mean(glm_preds == test_y)
# ---------------------------------------
train_lda <- train(train_x, train_y,
                   method = "lda")
lda_preds <- predict(train_lda, test_x)
mean(lda_preds == test_y)
# ---------------------------------------
train_qda <- train(train_x, train_y,
                   method = "qda")
qda_preds <- predict(train_qda, test_x)
mean(qda_preds == test_y)
# --------------------------------------
library(gam)
train_loess <- train(train_x, train_y, 
                     method = "gamLoess")
loess_preds <- predict(train_loess, test_x)
mean(loess_preds == test_y)
# --------------------------------------
set.seed(7,sample.kind = "Rounding")
train_knn <- train(train_x, train_y, 
                   method = "knn",
                   tuneGrid = data.frame(k = seq(3, 21, 2)))
knn_preds <- predict(train_knn, test_x)
mean(knn_preds == test_y)
train_knn$bestTune
# --------------------------------------
model <- c("kmeans_preds","glm_preds","lda_preds","qda_preds","rf_preds","knn_preds","loess_preds")
pred <- sapply(1:7, function(x){
  as.factor(get(model[x]))})
dim(pred)
# --------------------------------------
set.seed(9,sample.kind = "Rounding")
train_rf <- train(train_x, train_y, 
                  method = "rf",
                  tuneGrid = data.frame(mtry = c(3,5,7,9)))
rf_preds <- predict(train_rf, test_x)
mean(rf_preds == test_y)
train_rf$bestTune
varImp(train_rf)
# --------------------------------------
model <- c("kmeans_preds","glm_preds","lda_preds","qda_preds","rf_preds","knn_preds","loess_preds")
pred <- sapply(1:7, function(x){
  as.factor(get(model[x]))})
dim(pred)
# ---------------------------------------
pred <- as.data.frame(pred)
names(pred) <-c("kmeans_preds","glm_preds","lda_preds","qda_preds","rf_preds","knn_preds","loess_preds")
acc <- colMeans(as.matrix(pred)==test_y)
acc
# ---------------------------------------
ensemble <- cbind(glm = glm_preds == "B", lda = lda_preds == "B", qda = qda_preds == "B", loess = loess_preds == "B", rf = rf_preds == "B", knn = knn_preds == "B", kmeans = kmeans_preds == "B")

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "B", "M")
mean(ensemble_preds == test_y)