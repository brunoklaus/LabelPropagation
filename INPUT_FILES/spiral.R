library(mlbench)
setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/INPUT_FILES/")
# 1 cycle each, no noise
p<-mlbench.spirals(1500,sd = 0.04,cycles = 1)
p <- mlbench.circle(n =  1500)
p$x[,1] <- runif(n = nrow(p$x),min = -1,max=1)
p$x[,2] <- runif(n = nrow(p$x),min = -1,max=1)
#circle <- mlbench.circle(n =  1000)
#p$x <- rbind(p$x,  circle$x)
#p$classes <- c(p$classes,  circle$classes)
plot(p)
plot(circle)

#write.table(x = p$x,
#            file = "./benchmark/SSL,set=12,X.tab",sep='\t',
#            col.names=FALSE,row.names = FALSE)
#write.table(x = -1 + as.numeric(as.character(p$classes)),
#            file = "./benchmark/SSL,set=12,y.tab",
#            sep='\t',col.names = FALSE,row.names = FALSE)


setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/INPUT_FILES/")
p2 <- mlbench.spirals(1500,sd = 0.075,cycles = 1)
p2$x <- read.table(file = "./benchmark/SSL,set=20,X.tab")
setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/OUTPUT_FILES/5")
v <- vector(length = 9)
for (i in 0:9) {
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


setwd("/home/klaus/Downloads/eclipse-workspace/AAED_EIGEN/OUTPUT_FILES/4")
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

p2 <- mlbench.spirals(1500,sd = 0.075,cycles = 1)



p2$x <- read.table(file = "./benchmark/SSL,set=12,X.tab")
p2$classes <- as.vector(read.table(file = "./benchmark/SSL,set=12,y.tab"))
p2$classes <- as.factor(unname(unlist(p2$classes)))
df <- data.frame(x=p2$x[,1], y=p2$x[,2], class = p2$classes )
df$class <- as.character(df$class)

p1Id <- which.min(apply(df[,1:2],1,
                    function(x){
                      d1 <- x[1] - (-0.325)
                      d2 <- x[2] - (0.05)
                      return( sqrt(d1*d1+d2*d2) )
                      } ))
p2Id <- which.min(apply(df[,1:2],1,
                        function(x){
                          d1 <- x[1] - (0)
                          d2 <- x[2] - (0.5)
                          return( sqrt(d1*d1+d2*d2) )
                        } ))
df$class[which(1:nrow(df) %in% c(p1Id,p2Id) == FALSE)] <- rep(3,nrow(df)-2)



library(ggplot2)

ggplot() + 
  geom_point(data = df[which(df$class==3),], mapping = aes(x,y),color="forestgreen",shape=16) +
  geom_point(data = df[which(df$class==0),], mapping = aes(x,y),color="blue",fill="blue",shape=22,size=8) +
  geom_point(data = df[which(df$class==1),], mapping = aes(x,y),color="red",shape=17,size=8) 
  



library(class)
knnPred <- knn(train = df[which(1:nrow(df) %in% c(p1Id,p2Id) == TRUE),1:2],
    test= df[which(1:nrow(df) %in% c(p1Id,p2Id) == FALSE),1:2],
    cl = df[which(1:nrow(df) %in% c(p1Id,p2Id) == TRUE),3],
    k = 1)

df$class[which(1:nrow(df) %in% c(p1Id,p2Id) == FALSE)] <- knnPred

ggplot() + 
  geom_point(data = df[which(df$class==0),], mapping = aes(x,y),color="blue",fill="blue",shape=22) +
  geom_point(data = df[which(df$class==1),], mapping = aes(x,y),color="red",shape=17) 


peq.x <- matrix(data =
               c( 0,1,
                  0,0.5,
                  0,0
                 ),
               ncol = 2,byrow = TRUE)
for (i in 1.0:100.0) {
  theta <- i*2*pi/100.0 
  
  peq.x <- rbind(peq.x, c(0.001*cos(theta),- 0.001*sin(theta)))
}

peq.y <-c(1,1,0,rep(0,100))
write.table(x = peq.x,
                        file = "./benchmark/SSL,set=20,X.tab",sep='\t',
                        col.names=FALSE,row.names = FALSE)
write.table(x = peq.y,
            file = "./benchmark/SSL,set=20,y.tab",sep='\t',
            col.names=FALSE,row.names = FALSE)
