library("LogConcDEAD")
library("R.matlab")
library("R.utils")
library("mclust")

pathname <- "~/yEM.mat"
data <- readMat(pathname)

# Cule EM approach
timeCurr <- system.time(
{
	lcd <- EMmixlcd(data$DP,k=data$classes,verbose = 2)
})

writeMat(paste("~/yEMResult-",data$dataName,"-",dim(data$DP)[2],"D.mat",sep=""),logLikeCule = lcd$lcdloglik, logf = lcd$logf,props = lcd$props, niter = lcd$niter, timeReqCule=timeCurr[3])

