library(surveillance)

###################################################
### chunk number 1:
###################################################


# Implementation of the Bayes system.
# The system evaluates specified timepoints and gives alarm if it recognizes
# an outbreak for this timepoint.
#
# Features:
# Choice between different Bayes sub-systems (difference in reference values).

bayesScoreLatestTimepoint <- function(disProgObj, timePoint = NULL, control = list(b = 0, w = 6, actY = TRUE, alpha=0.05)){
  
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
  
  # construct the reference values
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
  
  # get the parameter for the negative binomial distribution
  # Modification on 13 July 2009 after comment by C. W. Ryan on NAs in the
  # time series
  sumBasevec <- sum(basevec, na.rm=TRUE)
  lengthBasevec <- sum(!is.na(basevec))
  
  # compute the upper limit of a one sided (1-alpha)*100% prediction interval.
  upPI <- qnbinom(1-control$alpha, sumBasevec + 1/2, (lengthBasevec)/(lengthBasevec + 1))
  
  # give alarm if the actual value is larger than the upper limit.
  alarm <- observed[timePoint] > upPI
  
  #################
  # MODIFY scores #
  #################
  score <- pnbinom(q=observed[timePoint], size=sumBasevec + 1/2, (lengthBasevec)/(lengthBasevec + 1))
  
  return(list(alarm=alarm, upperbound=upPI, score=score))
}


getBayes1Score <- function(data, freq){
  return(getBayesScore(data, freq, b=0, w=6, actY=TRUE, alpha=0.05))
}
getBayes2Score <- function(data, freq){
  return(getBayesScore(data, freq, b=1, w=6, actY=TRUE, alpha=0.05))
}
getBayes3Score <- function(data, freq){
  return(getBayesScore(data, freq, b=2, w=4, actY=FALSE, alpha=0.05))
}


getBayesScore <- function(data, freq, b=0, w=6, actY=TRUE, alpha=0.05) {
  
  disProgObj <- list(observed=data, freq=freq)
  range <- (w+1+b*freq):length(data)
  control <- list(range=range, b=b, w=w, alpha=alpha)
  
  ####################
  # Check parameters #
  # Default: Bayes 1 #
  ####################
  if(is.null(control$b)){
    control$b <- 0
  }
  if(is.null(control$w)){
    control$w <- 6
  }
  if(is.null(control$alpha)){
    control$alpha <- 0.05
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
    # call algo.bayesLatestTimepoint
    result <- bayesScoreLatestTimepoint(disProgObj, i, control = control)
    # store the results in the right order
    alarm[count] <- result$alarm
    upperbound[count] <- result$upperbound
    scores[count] <- result$score
    count <- count + 1
  }
  #Add name and data name to control object.
  control$name <- paste("bayes(",control$w,",",control$w*control$actY,",",control$b,")",sep="")
  control$data <- paste(deparse(substitute(disProgObj)))
  
  
  return(list(alarm=alarm, upperbound=upperbound, scores=scores))
}

