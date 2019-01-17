library("LogConcDEAD")
library("R.matlab")
library("R.utils")

printf <- function(...) cat(sprintf(...))

args <- commandArgs(trailingOnly = TRUE)
k = strtoi(args[1])
j = strtoi(args[2])
pathname <- paste("matfiles/timingsHellingerNew-",k,"-A.mat",sep="")
data <- readMat(pathname)
X = data$saveAll[[k]][[1]][[1]]
for (i in ((j-1)*9+1):(j*9+min(0,-(j-2)))) {
	timeCurr <- system.time(
	{  
		printf('k = %d, j = %d, i = %d\n',k,j,i)
		#lcd <- mlelcd(X[[i]][[1]],verbose=100)
	})
	#writeMat(paste("matfiles/results-",k,"-",j,"-",i,".mat",sep=""), triangulation = lcd$triang, logLikeCule = lcd$logMLE, b = lcd$b, beta = lcd$beta, timeReqCule=timeCurr[3], T=lcd$triang, XCule=lcd$x)
}
