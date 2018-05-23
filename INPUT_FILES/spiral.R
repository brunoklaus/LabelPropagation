library(mlbench)
setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/INPUT_FILES/")
# 1 cycle each, no noise
p<-mlbench.spirals(500,sd = 0.075,cycles = 1)
plot(p)

write.table(x = p$x,file = "./benchmark/SSL,set=0,X.tab",sep='\t',col.names=FALSE,row.names = FALSE)
write.table(x = -1 + as.numeric(as.character(p$classes)),file = "./benchmark/SSL,set=0,y.tab",
            sep='\t',col.names = FALSE,row.names = FALSE)




v <- vector(length = 9)
for (i in 0:100) {
  res <- data.frame(read.table(paste0("result_",i,".txt",collapse="")))
  p2 <- p
  p2$x <- p$x
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


res <- data.frame(read.table("results.txt"))
p2 <- p
p2$x <- p$x
p2$classes <- apply(res,1,function(x){
  y <- which.max(x)
  if (sort(x)[1] == sort(x)[2]) {
    return(length(x) + 1)
  }
  return( unname(y) ) }
)
plot(p2)