rm(list=ls())

source("../../utils/source_me.R", chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)
Global.PlotOpts$Prefix <- "../writeup/"

require(doSNOW)
require(snowfall)
require(parallel)

if (!sfIsRunning()) {
  sfInit(cpus=detectCores(), parallel = T)
  registerDoSNOW(sfGetCluster())
}

# We seem to use this color palette everywhere, just define it once
im.col <- gray((0:255)/255)

# Load Data ###################################################################
dat.raw <- LoadCacheTagOrRun("raw", read.csv, "../data/training.csv",
                             stringsAsFactors=F)
im.raw <- LoadCacheTagOrRun(
  "raw_im", sfLapply,
  dat.raw[,"Image"], function(x) {
    return(as.integer(unlist(strsplit(x, " "))))
  })
dat.raw$Image <- NULL
im.raw <- do.call(rbind, im.raw)

# Define some Image Utilities ----
RotateImage180 <- function(im) {
  return(apply(apply(im, 1, rev), 1, rev))
}
RotateKP180 <- function(kp) {
  return(96-kp)
}

ShowImage <- function(im) {
  # Shows the image with no axes and squared off.
  image(im, col=im.col, asp=1, axes=F)
}

ShowRotatedImage <- function(im) {
  ShowImage(RotateImage180(im))
}

TestRotations <- function() {
  # Makes an "F."  
  test_image <- matrix(0, 96, 96)
  # F part.
  test_image[81:85,11:85] <- 1
  test_image[11:79,11:15] <- 1
  test_image[11:79,41:45] <- 1
  # Dot part.
  test_image[11:15, 81:85] <- 1
  par(mfrow=c(1,2), mar=c(0,0,0,0))
  # A point in the center of the dot.
  kp_x = 12
  kp_y = 83
  ShowImage(test_image)
  points(kp_x/96, kp_y/96, col='red', pch=19)
  ShowImage(RotateImage180(test_image))
  points(RotateKP180(kp_x)/96, RotateKP180(kp_y)/96, col='red', pch=19)
}
#TestRotations()

# Show some random faces ######################################################
set.seed(0x0DedBeef)
num.faces <- 6
rand.idx <- sample(1:nrow(dat.raw), num.faces)
PlotSetup("random_faces")
par(mfrow=c(2,3), mar = c(0,0,0,0))
for (i in 1:num.faces) {
  kpx <- dat.raw[rand.idx[i], seq(1, ncol(dat.raw), 2)]
  kpy <- dat.raw[rand.idx[i], seq(2, ncol(dat.raw), 2)]
  
  #   image(matrix(im.raw[rand.idx[i],], 96, 96), asp=1, axes=F, col=im.col)
  #   points(kpx/96, kpy/96, col='red', pch='+')
  ShowRotatedImage(matrix(im.raw[rand.idx[i],], 96, 96))
  points(RotateKP180(kpx)/96, RotateKP180(kpy)/96, col='red', pch='+')
}
PlotDone()

# Visualize accuracy ##########################################################

run_dir = "run_e1000_swapaxes"

# Average face
avg.face <- matrix(colMeans(im.raw, na.rm=T), 96, 96)

# Average keypoints
avg.kp <- colMeans(dat.raw, na.rm=T)
avg.kpx <- avg.kp[seq(1, length(avg.kp), 2)]
avg.kpy <- avg.kp[seq(2, length(avg.kp), 2)]

features <- unique(gsub("_x|_y", "", names(dat.raw)))
feature.groups <- c("eyebrow", "eye_center", "eye_corner", "mouth_ex_bottom", 
                    "mouth_inc_bottom", "nose")

valid.pred <- read.csv(paste(run_dir, "combined_valid_pred.csv", sep="/"))
valid.actual <- read.csv(paste(run_dir, "combined_valid_actual.csv", sep="/"))

rmse <- sapply(names(dat.raw), function(f) {
  idx.pred <- (valid.pred[, paste("missing", gsub("_x|_y", "", f), sep="_")] < 0.5) &
    (valid.actual[, paste("missing", gsub("_x|_y", "", f), sep="_")] == 0)
  y <- valid.actual[idx.pred, f]
  yhat <- valid.pred[idx.pred, f]
  return(sqrt(mean((y-yhat)^2, na.rm=T)))
})

radius <- sapply(features, function(f) {
  idx.pred <- (valid.pred[, paste("missing", f, sep="_")] < 0.5) &
    (valid.actual[, paste("missing", f, sep="_")] == 0)
  x <- valid.actual[idx.pred, paste(f, "x", sep="_")]
  xhat <- valid.pred[idx.pred, paste(f, "x", sep="_")]
  y <- valid.actual[idx.pred, paste(f, "y", sep="_")]
  yhat <- valid.pred[idx.pred, paste(f, "y", sep="_")]
  radius <- mean(sqrt((x-xhat)^2+(y-yhat)^2))
  return(radius)
})

# Colors
pal <- gg_color_hue(6)
pal.light <- add.alpha(pal, alpha=0.2)

# Get feature groups
require(jsonlite)
fgps <- fromJSON("feature_groups.json")

# Plot
PlotSetup("avg_face_rmse")
# image(avg.face, col = im.col, xaxt='n', yaxt='n')
ShowRotatedImage(avg.face)
for (i in 1:length(fgps)) {
  fn <- unique(gsub("_x|_y", "", names(dat.raw)[fgps[[i]]+1]))
  cat(fn, "\n\n")
  #   points(avg.kpx[paste(fn, "x", sep="_")]/96, 
  #          avg.kpy[paste(fn, "y", sep="_")]/96, col=pal[i], pch='+')  
  #   symbols(avg.kpx[paste(fn, "x", sep="_")]/96, 
  #           avg.kpy[paste(fn, "y", sep="_")]/96, 
  #           circles=radius[fn]/96, 
  #           fg=pal[i], bg=pal.light[i],
  #           inches=F, add=T)
  points(RotateKP180(avg.kpx[paste(fn, "x", sep="_")])/96, 
         RotateKP180(avg.kpy[paste(fn, "y", sep="_")])/96,
         col=pal[i], pch='+')  
  symbols(RotateKP180(avg.kpx[paste(fn, "x", sep="_")])/96, 
          RotateKP180(avg.kpy[paste(fn, "y", sep="_")])/96, 
          circles=radius[fn]/96, 
          fg=pal[i], bg=pal.light[i],
          inches=F, add=T)
}
PlotDone()

# GOOD AND BAD PREDICTIONS ####################################################

img.rmse <- data.frame(index = valid.pred$index, rmse = rep(NA, nrow(valid.pred)))
img.rmse$rmse <- sapply(1:nrow(valid.pred), function(i) {
  y <- valid.actual[i,2:31]
  yhat <- valid.pred[i,2:31]
  return(sqrt(mean((y-yhat)^2, na.rm=T)))
})

# Restrict to only images where all keypoints are actually there
allthere.idx <- valid.actual$index[rowSums(
  valid.actual[, grep("missing", names(valid.actual))]
) == 0]
img.rmse <- img.rmse[img.rmse$index %in% allthere.idx,]

PlotFaces <- function(indices) {
  for (idx in indices) {
    # Plot face
    #   image(matrix(im.raw[idx+1,], 96, 96), 
    #         col = im.col, xaxt='n', yaxt='n')
    ShowRotatedImage(matrix(im.raw[idx+1,], 96, 96))
    
    # Predicted keypoints
    kpx.pred <- valid.pred[valid.pred$index==idx, grep("_x", names(valid.pred))]
    kpy.pred <- valid.pred[valid.pred$index==idx, grep("_y", names(valid.pred))]
    
    # Actual keypoints
    kpx.actual <- valid.actual[valid.actual$index==idx, grep("_x", names(valid.actual))]
    kpy.actual <- valid.actual[valid.actual$index==idx, grep("_y", names(valid.actual))]
    
    #   points(kpx.pred/96, kpy.pred/96, col='red', pch='+')
    #   points(kpx.actual/96, kpy.actual/96, col='green', pch='o')
    points(RotateKP180(kpx.pred)/96, RotateKP180(kpy.pred)/96, col='red', pch='+')
    points(RotateKP180(kpx.actual)/96, RotateKP180(kpy.actual)/96, col='green', pch='o')
  }
}

# Order from most to least accurate
img.rmse <- img.rmse[order(img.rmse$rmse), ]
PlotSetup("best_faces")
par(mfrow=c(2, 3), mar=rep_len(0, 4))
PlotFaces(img.rmse$index[1:6])
PlotDone()

# Order from least to most accurate
img.rmse <- img.rmse[order(img.rmse$rmse, decreasing = T), ]
PlotSetup("worst_faces")
par(mfrow=c(2, 3), mar=rep_len(0, 4))
PlotFaces(img.rmse$index[1:6])
PlotDone()

# Look for off-by-one
# 
# test.idx <- 100
# test.row <- as.matrix(valid.actual[test.idx, 2:31])
# diff <- sapply(1:nrow(dat.raw), function(r) {
#   return(sum(abs(as.matrix(dat.raw[r,]) - test.row), na.rm=T))
# })
# which.min(diff)-1
# valid.actual$index[test.idx]
# 
# 