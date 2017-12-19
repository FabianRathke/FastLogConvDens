library("LogConcDEAD")
library("R.matlab")
library("R.utils")


args <- commandArgs(trailingOnly = TRUE)
pathname <- paste("~/",args[1],".mat",sep="")
data <- readMat(pathname)
timeCurr <- system.time(
{  
	lcd <- mlelcd(data$X,verbose=1)
})
writeMat(paste("~/",data$filename,".mat",sep=""), triangulation = lcd$triang, logLikeCule = lcd$logMLE, b = lcd$b, beta = lcd$beta, timeReqCule=timeCurr[3], T=lcd$triang, XCule=lcd$x)
