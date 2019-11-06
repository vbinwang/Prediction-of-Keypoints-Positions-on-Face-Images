rm(list=ls())

require(reshape2)
require(ggplot2)

PlotLoss <- function(file, rx='^train_rmse') {
    dat <- read.csv(file)
    loss <- dat[, grep(rx, names(dat))]
    loss <- data.frame(epoch=dat$epoch, loss)
    melted <- melt(loss, id.vars = 'epoch')
    
    g <- ggplot(data=melted) + 
      geom_line(aes(x=epoch, y=value, color=variable)) +
      scale_y_log10()
    plot(g) 
}

PlotValidationLoss <- function(file, excl='^train') {
  dat <- read.csv(file)
  rmse <- dat[, -grep(excl, names(dat))]
  melted <- melt(rmse, id.vars='epoch')

  g <- ggplot(data=melted) +
    geom_line(aes(x=epoch, y=value, color=variable)) +
    scale_y_log10()
  plot(g)
}

PlotEverything <- function(file) {
  dat <- read.csv(file)
  melted <- melt(dat, id.vars='epoch')
  g <- ggplot(data=melted) +
    geom_line(aes(x=epoch, y=value, color=variable)) +
    scale_y_log10()
  plot(g)
}
