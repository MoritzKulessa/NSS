library(surveillance)

### R code from vignette source 'Rnw/algo_rki.Rnw'
### Encoding: ISO8859-1

###################################################
### code chunk number 1: algo_rki.Rnw:96-214
###################################################


# Implementation of the Robert-Koch Institute (RKI) surveillance system.
# The system evaluates specified timepoints and gives alarm if it recognizes
# an outbreak for this timepoint.
#
# Features:
# Choice between the different RKI sub-systems (difference in reference values).

rkiScoreLatestTimepoint <- function(disProgObj, timePoint=NULL, control=list(b=0, w=6, actY=TRUE)){
  
  observed <- disProgObj$observed
  freq <- disProgObj$freq
  
  # If there is no value in timePoint, then take the last value in observed
  if(is.null(timePoint)){
    timePoint = length(observed)
  }
  
  # check if the vector observed includes all necessary data.
  if((timePoint-(control$b*freq)-control$w) < 1){
    stop("The vector of observed is too short!")
  }
  
  # Extract the reference values from the historic time series
  basevec <- c()
  # if actY == TRUE use also the values of the year of timepoint
  if(control$actY){
    basevec <- observed[(timePoint - control$w):(timePoint - 1)]
  }
  # check if you need more referencevalues of the past
  if(control$b >= 1){
    for(i in 1:control$b){
      basevec <- c(basevec, observed[(timePoint-(i*freq)-control$w):(timePoint-(i*freq)+control$w)])
    }
  }
  
  # compute the mean.
  mu <- mean(basevec)

  score <- NA

  if(mu > 20){ # use the normal distribution.
    # comupte the standard deviation.
    sigma <- sqrt(var(basevec))
    # compute the upper limit of the 95% CI.
    upCi <- mu + 2 * sigma
    
    #################
    # MODIFY scores #
    #################
    score <- pnorm(observed[timePoint], mean=mean(basevec), sd=sd(basevec))
    
  }
  else{ # use the poisson distribution.
    # take the upper limit of the 95% CI from the table CIdata.txt.
    #data("CIdata", envir=environment())   # only local assignment -> SM: however, should not use data() here
    #CIdata <- read.table(system.file("data", "CIdata.txt", package="surveillance"), header=TRUE)
    #SM: still better: use R/sysdata.rda (internal datasets being lazy-loaded into the namespace environment)
    # for the table-lookup mu must be rounded down.
    mu <- floor(mu)
    # we need the third column in the row mu + 1
    
    
    CIdata <- c(3.285, 5.323, 6.686, 8.102, 9.598, 11.177, 12.817, 13.765, 14.921, 16.768, 17.633, 19.050, 20.335, 21.364, 22.945, 23.762, 25.400, 26.306, 27.735, 28.966, 30.017)
    
    upCi <- CIdata[mu + 1]
    
    
    #################
    # MODIFY scores #
    #################
	score <- ppois(observed[timePoint], floor(mean(basevec))+1)
    

    
  }
  # give alarm if the actual value is larger than the upper limit.
  alarm <- observed[timePoint] > upCi
  
  return(list(alarm=alarm, upperbound=upCi, score=score))
}

# 'algo.rki' calls 'algo.bayesLatestTimepoint' for data points given by range.

getRKIScore <- function(data, freq, b=0, w=6, actY=TRUE) {  
  
  disProgObj <- list(observed=data, freq=freq)
  range <- (w+1+b*freq):length(data)
  control <- list(range=range, b=b, w=w)
  
  # Set the default values if not yet set
  if(is.null(control$b)){
    control$b <- 0
  }
  if(is.null(control$w)){
    control$w <- 6
  }
  if(is.null(control$actY)){
    control$actY <- TRUE
  }
  
  ####################################
  # initialize the necessary vectors #
  ####################################
  alarm <- matrix(data = 0, nrow = length(control$range), ncol = 1)
  upperbound <- matrix(data = 0, nrow = length(control$range), ncol = 1)
  scores <- matrix(data = 0, nrow = length(control$range), ncol = 1)
  
  count <- 1
  for(i in control$range){

    result <- rkiScoreLatestTimepoint(disProgObj, i, control = control)
    
    # store the results in the right order
    alarm[count] <- result$alarm
    upperbound[count] <- result$upperbound
    scores[count] <- result$score
    count <- count + 1
  }
  
  return(list(alarm=alarm, upperbound=upperbound, scores=scores))
}

getRKI1Score <- function(data, freq) {
  return(getRKIScore(data, freq, b=0, w=6, actY=TRUE))
}
getRKI2Score <- function(disProgObj, control = list(range = range)){
  return(getRKIScore(data, freq, b=1, w=6, actY=TRUE))
}
getRKI3Score <- function(disProgObj, control = list(range = range)){
  return(getRKIScore(data, freq, b=2, w=4, actY=FALSE))
}
