library("LogConcDEAD")
library("R.matlab")
library("R.utils")
library("mclust")

pathname <- "~/yEM.mat"
data <- readMat(pathname)

# Cule EM approach
#lcd <- EMmixlcd(data$DP)
#writeMat("~/yEMResult.mat",logLikeCule = lcd$lcdloglik, logf = lcd$logf,props = lcd$props, niter = lcd$niter)

# Hierarchical Clustering used by Cule as initialization
k <- data$classes
x <- as.matrix( data$DP )
n <- nrow(x)
d <- ncol(x)
highclust <- hc( modelName="VVV", data$DP )
class <- c( hclass( highclust, k ) )
props <- rep( 0, k )
y <- matrix( 0, nrow=n, ncol=k )
for( i in 1:k ) {
  props[ i ] <- sum( class==i ) / n
  ss <- x[ class==i, ]
  y[ , i ] <- dmvnorm( x, mean=apply( ss,2, mean), sigma=var( ss ), log=TRUE )
}

pif <- t( t( exp( y ) ) * props )
theta <- pif / apply( pif, 1, sum )

writeMat("~/yEMInit.mat", props = props, class = class, posterior = theta, y=y)

#oldloglik <- sum( log( apply( apply( exp( y ), 1, "*", props ), 2, sum)  ) )
