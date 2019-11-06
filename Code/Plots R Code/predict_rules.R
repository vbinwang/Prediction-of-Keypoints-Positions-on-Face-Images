rm(list=ls())
source('../../utils/source_me.R', chdir=T)
CreateDefaultPlotOpts()
Global.PlotOpts$Prefix <- "../writeup/"

dir <- 'run_e1000_swapaxes/all_binary'

# Feature Names ----
CapLeading <- function (string){
  fn <- function(x){
    v <- unlist(strsplit(x, split = " "))
    u <- sapply(v, function(x){
      x <- tolower(x)
      substring(x, 1, 1) <- toupper(substring(x, 1, 1))
      x})
    paste(u, collapse = " ")
  }
  sapply(string, fn)
}

feature_names <- c(
  "left_eye_center", "right_eye_center", 
  "left_eye_inner_corner", "right_eye_inner_corner",
  "left_eye_outer_corner", "right_eye_outer_corner",
  "left_eyebrow_inner_end", "right_eyebrow_inner_end",
  "left_eyebrow_outer_end", "right_eyebrow_outer_end",
  "nose_tip", "mouth_left_corner", "mouth_right_corner",
  "mouth_center_top_lip", "mouth_center_bottom_lip")

require(stringr)
pretty_feature_names <- unname(sapply(feature_names, function(x) 
  CapLeading(str_replace_all(x, "_", " "))))

missing_feature_names <- paste("missing", feature_names, sep="_")

# Load Y and Y-hat ----
train_y_hat <- read.csv(file.path(dir, 'last_layer_train.csv'), row.names='index')
train_y <- read.csv(file.path(dir, 'y_train.csv'), row.names='index')

require(ggplot2)

logistic_boxplots <- lapply(1:length(feature_names), function(i) {
  col_name = missing_feature_names[i]
  print(sprintf("%s: %s", pretty_feature_names[i], col_name))
  df <- data.frame(
    y_hat=train_y_hat[, col_name], 
    y=factor(train_y[, col_name], levels=c(0,1), labels=c("Present", "Missing")))
  
  
  g <- ggplot() + geom_boxplot(data=df, aes(factor(y), y_hat, fill=factor(y))) +
    guides(fill=F) + theme_bw() + 
    xlab('Actual') + 
    ylab('Pr(Missing)') +
    ggtitle(pretty_feature_names[i])
  return(g)
})
eyes <- logistic_boxplots[1:6]
eyebrow <- logistic_boxplots[7:10]
mouth <- logistic_boxplots[12:15]

PlotSetup('logistic_boxplots_eye')
MultiPlot(plotlist=eyes, layout=matrix(1:6, nrow=3, byrow=T))
PlotDone()

PlotSetup('logistic_boxplots_eyebrow')
MultiPlot(plotlist=eyebrow, layout=matrix(1:4, nrow=2, byrow=T))
PlotDone()

PlotSetup('logistic_boxplots_mouth')
MultiPlot(plotlist=mouth, layout=matrix(1:4, nrow=2, byrow=T))
PlotDone()

PlotSetup('logistic_boxplots_nose')
plot(logistic_boxplots[[11]])
PlotDone()

# Determine Optimal Cutoffs ----
require(OptimalCutpoints)
cutoffs <- lapply(missing_feature_names, function(col_name, cutoff_method="MaxProdSpSe") {
  print(col_name)
  df = data.frame(
    y_hat=train_y_hat[,col_name],
    y=train_y[,col_name])
  if (all(df$y == 1) || all(df$y == 0)) {
    return(list(cutoff=0.5, obj=NULL))
  }
  
  opt <- optimal.cutpoints("y_hat", "y", methods=cutoff_method, 
                           data=df, tag.healthy=0)
  cutoff <- opt[['ROC01']][['Global']][['optimal.cutoff']][['cutoff']]
  return(list(cutoff=cutoff, obj=opt))
})

cutoff_table <- data.frame(
  feature_name=missing_feature_names,
  Feature=pretty_feature_names,
  OptimalCutoff=unlist(sapply(cutoffs, function(x) x[['cutoff']])),
  row.names="feature_name")

# Plot ROC Curve ----
# Find an interesting ROC curve to Plot.
PlotSetup(sprintf("roc_%s", feature_names[1]))
layout(matrix(1:2, ncol=2))
plot(cutoffs[[1]][['obj']])
PlotDone()

PlotSetup(sprintf("roc_%s", feature_names[3]))
layout(matrix(1:2, ncol=2))
plot(cutoffs[[3]][['obj']])
PlotDone()

# Training Confusion Matrices ----
require(caret)

getConfusionMatrices <- function(cutoff_table, y_hat, y) {
  return(lapply(missing_feature_names, function(name) {
    if (length(levels(factor(y[,name]))) < 2) {
      return(NULL)
    }
    hard_pred <- as.numeric(
      y_hat[, name] > cutoff_table[name, "OptimalCutoff"])
    return(confusionMatrix(hard_pred, y[, name], positive='1'))
  }))
}

train_confuse <- getConfusionMatrices(cutoff_table, train_y_hat, train_y)

# Validation Confusion Matrices ----
valid_y_hat <- read.csv(file.path(dir, 'last_layer_val.csv'), row.names='index')
valid_y <- read.csv(file.path(dir, 'y_validate.csv'), row.names='index')

valid_confuse <- getConfusionMatrices(cutoff_table, valid_y_hat, valid_y)

valid_table <- data.frame(
      Feature=pretty_feature_names,
      Cutoff=cutoff_table$OptimalCutoff,
      t(sapply(valid_confuse, function(confuse) {
        if (is.null(confuse)) {
          return(list(Sensitivity=NA, Specificity=NA))
        }
        list(Sensitivity=confuse[['byClass']][['Sensitivity']],
             Specificity=confuse[['byClass']][['Specificity']])
      })))
ExportTable(valid_table, 'logistic_cutoff_table', 
            'Cutoff Validation Performance', digits=3, include.rownames=F)
