setwd("C:/Users/Administrator/Downloads/CoastSat/data/POYANG/jpg_files")
getwd()
mydirs <- list.dirs(getwd())
getthis <- "name"

for(i in 2:length(mydirs)){
  thejpgs <- list.files(mydirs[[i]])
  ind <- which(thejpgs == getthis, arr.ind = TRUE)
  new_name <- strsplit(mydirs[[i]], "/")[[1]][9]
  keep <- thejpgs[ind]
  name(keep) <- new_name
}
