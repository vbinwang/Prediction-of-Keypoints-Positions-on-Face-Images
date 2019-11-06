rm(list=ls())

dat <- read.csv('../data/training.csv')
dat <- dat[,1:30]

nacols <- sapply(dat, function(x) { round(sum(is.na(x))/10)*10 } )

eyecenterIdx <- grep("eye_center", names(dat))
eyecenterNaN <- sum(is.na(rowSums(dat[, eyecenterIdx])))

eyecornerIdx <- grep(glob2rx("*eye_*_corner*"), names(dat))
eyecornerNaN <- sum(is.na(rowSums(dat[, eyecornerIdx])))

eyebrowIdx <- grep(glob2rx("*eyebrow*"), names(dat))
eyebrowNaN <- sum(is.na(rowSums(dat[, eyebrowIdx])))

mouthExBottomIdx <- c(grep(glob2rx("mouth_*_corner*"), names(dat)), 
                      grep(glob2rx("mouth_center_top*"), names(dat)))
mouthExBottomNaN <- sum(is.na(rowSums(dat[, mouthExBottomIdx])))

mouthBottomIdx <- grep(glob2rx("mouth_*_bottom_*"), names(dat))
mouthBottomNaN <- sum(is.na(rowSums(dat[, mouthBottomIdx])))

noseIdx <- grep(glob2rx("nose_tip_*"), names(dat))
noseNaN <- sum(is.na(rowSums(dat[, noseIdx])))

# Did I get all the columns?
length(eyecenterIdx) + length(eyecornerIdx) + length(eyebrowIdx) + 
  length(mouthBottomIdx) + length(mouthExBottomIdx) + length(noseIdx) == length(nacols)

groupNaN <- c(rep(eyecenterNaN, 4), rep(eyecornerNaN, 8), rep(eyebrowNaN, 8), rep(noseNaN, 2),
            rep(mouthExBottomNaN, 6), rep(mouthBottomNaN, 2))

# How many do we drop?
dropShare <- 100 * (groupNaN - nacols) / (nrow(dat) - nacols)
dropShare[order(dropShare, decreasing = T)]

# Print (remember that Python 0-indexes and R 1-indexes)
require(jsonlite)

featuregroups <- list()
featuregroups[['eye_center']] <- eyecenterIdx - 1
featuregroups[['eye_corner']] <- eyecornerIdx - 1
featuregroups[['eyebrow']] <- eyebrowIdx - 1
featuregroups[['mouth_inc_bottom']] <- mouthBottomIdx - 1
featuregroups[['mouth_ex_bottom']] <- mouthExBottomIdx - 1
featuregroups[['nose']] <- noseIdx - 1
write(toJSON(featuregroups, pretty=T), 'feature_groups.json')
