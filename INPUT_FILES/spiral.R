library(mlbench)
setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/INPUT_FILES/")
# 1 cycle each, no noise
p<-mlbench.spirals(1500,sd = 0.075,cycles = 1)
p <- mlbench.circle(n =  1500)
p$x[,1] <- runif(n = nrow(p$x),min = -1,max=1)
p$x[,2] <- runif(n = nrow(p$x),min = -1,max=1)
#circle <- mlbench.circle(n =  1000)
#p$x <- rbind(p$x,  circle$x)
#p$classes <- c(p$classes,  circle$classes)
plot(p)
plot(circle)

write.table(x = p$x,
            file = "./benchmark/SSL,set=11,X.tab",sep='\t',
            col.names=FALSE,row.names = FALSE)
write.table(x = -1 + as.numeric(as.character(p$classes)),
            file = "./benchmark/SSL,set=11,y.tab",
            sep='\t',col.names = FALSE,row.names = FALSE)


setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/INPUT_FILES/")
p2 <- mlbench.spirals(1500,sd = 0.075,cycles = 1)
p2$x <- read.table(file = "./benchmark/SSL,set=0,X.tab")
setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/OUTPUT_FILES/2")
v <- vector(length = 9)
for (i in 0:99) {
  res <- data.frame(read.table(paste0("result_iter_",i,".txt",collapse="")))
  p2$classes <- apply(res,1,function(x){
    y <- which.max(x)
    if (sort(x)[1] == sort(x)[2]) {
      return(length(x) + 1)
    }
    return( unname(y) ) }
  )
  png(filename=paste0("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/pics/result_",i,".png",collapse=""))
  plot(p2)
  dev.off()
}
res <- data.frame(read.table("final.txt"))
p2$classes <- apply(res,1,function(x){
  y <- which.max(x)
  if (sort(x)[1] == sort(x)[2]) {
    return(length(x) + 1)
  }
  return( unname(y) ) }
)
png(filename="/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/pics/result_final.png")
plot(p2)
dev.off()


setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/OUTPUT_FILES/2")
res <- data.frame(read.table("set_0_run_0_samples_100_prediction.txt"))
p2 <- mlbench.spirals(1500,sd = 0.075,cycles = 1)

setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/INPUT_FILES/")
p2$x <- read.table(file = "./benchmark/SSL,set=0,X.tab")

p2$classes <- apply(res,1,function(x){
  y <- which.max(x)
  if (sort(x)[1] == sort(x)[2]) {
    return(length(x) + 1)
  }
  return( unname(y) ) }
)
plot(p2)
