library("logcondens")
library("R.matlab")
library("R.utils")
args <- commandArgs(trailingOnly = TRUE)
pathname <- paste("~/",args[1],".mat",sep="")
data <- readMat(pathname)
timeCurr <- system.time(
{ 
	for (i in 1:10){
		print(i)
		dlc <- logConDens(data$X,print=FALSE,smoothed=FALSE) 
	}
})

writeMat(paste("~/",data$filename,".mat",sep=""), knots = dlc$IsKnot, xRufi = dlc$x, timeReqRufi = timeCurr[3], phi = dlc$phi) 

#if (dim(data$X)[2] == 2) {
#	evalDens <- dlcd(matrix(c(data$XXX,data$YYY),ncol=2),lcd)
#	g <- interplcd(lcd, gridlen=100)
#	writeMat("~/yreturnND.mat", logLikeCule = lcd$logMLE, b = lcd$b, beta = lcd$beta, timeReqCule=timeCurr[3], gX = g$x, gY = g$y, gZ = g$z, evalDens = evalDens)
#} else {
#		writeMat("~/yreturnND.mat", logLikeCule = lcd$logMLE, b = lcd$b, beta = lcd$beta, timeReqCule=timeCurr[3])
#}



