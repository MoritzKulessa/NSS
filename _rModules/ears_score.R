library("surveillance")

#         \|||/
#         (o o)
# ,~~~ooO~~(_)~~~~~~~~~,
# |        EARS        |
# |   surveillance     |
# |       methods      |
# |   C1, C2 and C3    |
# | MODIFIED BY Moritz |
# '~~~~~~~~~~~~~~ooO~~~'
#        |__|__|
#         || ||
#        ooO Ooo


######################################################################
# Implementation of the EARS surveillance methods.
######################################################################
# DESCRIPTION
######################################################################
# Given a time series of disease counts per month/week/day
# this function determines whether there was an outbreak at given time points:
# it deduces for each time point an expected value from past values,
# it defines an upperbound based on this value and on the variability
# of past values
# and then it compares the observed value with the upperbound.
# If the observed value is greater than the upperbound
# then an alert is flagged.
# Three methods are implemented.
# They do not use the same amount of past data
# and are expected to have different specificity and sensibility
# from C1 to C3
# the amount of past data used increases,
# so does the sensibility
# but the specificity decreases.
######################################################################
# PARAMETERS
######################################################################
#   range : range of timepoints over which the function will look for
# outbreaks.
#   method : which of the three EARS methods C1, C2 and C3 should be used.
#
######################################################################
# INPUT
######################################################################
# A R object of class sts
######################################################################
# OUTPUT
######################################################################
# The same R object of class sts with slot alarm and upperbound filled
# by the function
######################################################################

getEarsScore <- function(data, freq, method="C1", baseline=7, alpha=0.05, minSigma=1) {
  
  sts <- sts(data, frequency= freq)
  control <- list(range = NULL, method = method, baseline = baseline, alpha = alpha, minSigma = minSigma)
  
  
  ######################################################################
  #Handle I/O
  ######################################################################
  #If list elements are empty fill them!
  if (is.null(control[["baseline", exact = TRUE]])) {
    control$baseline <- 7
  }
  
  if (is.null(control[["minSigma", exact = TRUE]])) {
    control$minSigma <- 0
  }
  baseline <- control$baseline
  minSigma <- control$minSigma
  
  if(minSigma < 0) {
    stop("The minimum sigma parameter (minSigma) needs to be positive")
  }
  if (baseline < 3) {
    stop("Minimum baseline to use is 3.")
  }
  
  # Method
  if (is.null(control[["method", exact = TRUE]])) {
    control$method <- "C1"
  }
  
  # Extracting the method
  method <- match.arg( control$method, c("C1","C2","C3","C4"),several.ok=FALSE)
  
  # Range
  # By default it will take all possible weeks
  # which is not the same depending on the method
  if (is.null(control[["range",exact=TRUE]])) {
    if (method == "C1"){
      control$range <- seq(from=baseline+1, to=dim(sts@observed)[1],by=1)
    }
    if (method == "C2"){
      control$range <- seq(from=baseline+3, to=dim(sts@observed)[1],by=1)
    }
    if (method == "C3"){
      control$range <- seq(from=baseline+5, to=dim(sts@observed)[1],by=1)
    }
  }
  
  
  
  # Calculating the threshold zAlpha
  zAlpha <- qnorm((1-control$alpha))
  
  
  #Deduce necessary amount of data from method
  maxLag <- switch(method, C1 = baseline, C2 = baseline+2, C3 = baseline+4, C4 = baseline+4)
  
  # Order range in case it was not given in the right order
  control$range = sort(control$range)
  
  
  #################
  # MODIFY scores #
  #################
  scores <- y<-matrix(NA, nrow=length(sts@observed),ncol=ncol(sts))
  
  ######################################################################
  #Loop over all columns in the sts object
  #Call the right EARS function depending on the method chosen (1, 2 or 3)
  #####################################################################
  for (j in 1:ncol(sts)) {
    
    # check if the vector observed includes all necessary data: maxLag values.
    if((control$range[1] - maxLag) < 1) {
      stop("The vector of observed is too short!")
    }
    
    ######################################################################
    # Method C1 or C2
    ######################################################################
    if(method == "C1"){
      # construct the matrix for calculations
      ndx <- as.vector(outer(control$range, baseline:1, FUN = "-"))
      refVals <- matrix(observed(sts)[,j][ndx], ncol = baseline)
      sts@upperbound[control$range, j] <- apply(refVals,1, mean) + zAlpha * pmax(apply(refVals, 1, sd), minSigma)
      
      #################
      # MODIFY scores #
      #################
      check_vals <- matrix(observed(sts)[control$range, ])[ ,j]
      means <- apply(refVals,1, mean)
      stdv <- pmax(apply(refVals, 1, sd), minSigma)
      for (idx in 1:length(means)){
        scores[control$range[idx],j] <- pnorm(check_vals[idx],mean=means[idx],sd=stdv[idx])
      }
      
    }
    
    if (method == "C2") {
      # construct the matrix for calculations
      ndx <- as.vector(outer(control$range, (baseline + 2):3, FUN = "-"))
      refVals <- matrix(observed(sts)[,j][ndx], ncol = baseline)
      sts@upperbound[control$range, j] <- apply(refVals,1, mean) + zAlpha * pmax(apply(refVals, 1, sd), minSigma)
      
      #################
      # MODIFY scores #
      #################
      check_vals <- matrix(observed(sts)[control$range, ])[ ,j]
      means <- apply(refVals,1, mean)
      stdv <- pmax(apply(refVals, 1, sd), minSigma)
      for (idx in 1:length(means)){
        scores[control$range[idx],j] <- pnorm(check_vals[idx],mean=means[idx],sd=stdv[idx])
      }
    }
    
    if (method == "C3") {
      
      rangeC2 = ((min(control$range) - 2):max(control$range))
      
      ##HB replacing loop:
      ndx <- as.vector(outer(rangeC2, (baseline + 2):3, FUN = "-"))
      refVals <- matrix(observed(sts)[,j][ndx], ncol = baseline)
      
      ##HB using argument 'minSigma' to avoid dividing by zero, huge zscores:
      C2 <- (observed(sts)[rangeC2, j] - apply(refVals, 1, mean))/pmax(apply(refVals, 1, sd), minSigma)
      
      partUpperboundLag2 <- pmax(rep(0, length = length(C2) - 2), C2[1:(length(C2) - 2)] - 1)
      partUpperboundLag1 <- pmax(rep(0, length = length(C2) - 2), C2[2:(length(C2) - 1)] - 1)
      
      ##HB using argument 'minSigma' to avoid alerting threshold that is zero or too small
      sts@upperbound[control$range, j] <- observed(sts)[control$range, j] +
        pmax(apply(as.matrix(refVals[3:length(C2), ]),1, sd),minSigma) *
        (zAlpha - (partUpperboundLag2 + partUpperboundLag1))
      
      sts@upperbound[control$range, j] = pmax(rep(0, length(control$range)), sts@upperbound[control$range, j])
	  
      #################
      # MODIFY scores #
      #################
      
      check_vals <- matrix(observed(sts)[rangeC2, ])[ ,j]
      means <- apply(refVals,1, mean)
      stdv <- pmax(apply(refVals, 1, sd), minSigma)
      for (idx in 3:length(means)){
        
        z_score_t2 <- pmax(0, ((check_vals[idx-2]-means[idx-2]) / stdv[idx-2])-1)
        z_score_t1 <- pmax(0, ((check_vals[idx-1]-means[idx-1]) / stdv[idx-1])-1)
		z_score_t0 <- ((check_vals[idx]-means[idx]) / stdv[idx])
        scores[rangeC2[idx],j] <- pnorm((z_score_t0 - z_score_t1 - z_score_t2),mean=0,sd=1)

      }
      
    }
    
    
  }
  
  #Copy administrative information
  control$name <- paste("EARS_", method, sep = "")
  control$data <- paste(deparse(substitute(sts)))
  sts@control <- control
  sts@alarm[control$range, ] <- matrix(observed(sts)[control$range, ] > upperbound(sts)[control$range, ])
  
  
  return(list(alarm=sts@alarm, upperbound=sts@upperbound, scores=scores))
  
}


