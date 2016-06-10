library(xtable)
library(RankAggreg)
require(PMCMR)


source("~/Documents/Master Degree/Master Project/Implementation/LazyNN_RF/experiments/reports/f1-measure.R")

splitAt <- function(x, pos) unname(split(x, cumsum(seq_along(x) %in% pos)))

parser_class_column <- function(column){
  return(as.numeric(sub(":[0-9]+.*[0-9]*","", sub("CLASS=", "", column))))
}

result.load <- function(file, trials, metric = "f1"){
  # Load a result file and compute an given metric.
  #
  # Args:
  #   file: An string contrain the path to file.
  #   trials: Number of trials performed in the experiment that resulted the file.
  #   metric: Metric to be computed. Default is "f1".
  #
  # Returns:
  #   Return a matrix contain the metric value for each trial.
  n_metric = 1
  if (metric == "f1") {
    n_metric = 2 
  }
  n_cols = seq_len(max(count.fields(file, sep = ' ')))

  lines <- readLines(file)
  splits <- grep("#", lines)
  splits[2:length(splits)] <- splits[2:length(splits)] - 1:(length(splits)-1) 
  
  table = read.table(file, header = F,  stringsAsFactors=FALSE,
                     fill = T, col.names = paste0("V", n_cols))
  #table = read.table(file, header = F,  fill = T,
  #                   colClasses=c("integer","integer","integer"),
  #                   stringsAsFactors=FALSE, comment.char="#")
  #splits <- which(table$V1 == 0)
  ids = parser_class_column(table$V1)
  y = parser_class_column(table$V2)
  pred = parser_class_column(table$V3)
  
  y <- splitAt(y, splits)
  pred <- splitAt(pred, splits)
  
  trials <- length(pred)
  
  results = array(0, c(trials, n_metric))
  
  for (j in 1:trials) {
    # Compute metric
    if (metric == "f1") {
      f1_meas = f1_measure(y[[j]] , pred[[j]])
      results[j,1] = f1_meas[[1]]
      results[j,2] = f1_meas[[2]]
    } else {
      stop("not supported metric has been chosen")
    }
  }
  return(results)
}

result.load.dir <- function(dir, trials, metric = "f1"){
  # Load result files from a given directory
  # and compute a given metric.
  #
  # Args:
  #   dir: Directory path. The filenames must follow the pattern
  #         results_${model-name}_${dataset-name}.
  #   trials: Number of trials performed in the experiment that resulted the files.
  #   metric: Metric to be computed. Default is "f1".
  #
  # Returns:
  #   Return a 3d matrix contain the metric(matrix dimensions are (model, dataset, trial)),
  #   and also return the model and dataset names.
  n_metric = 1
  if(metric == "f1"){
    n_metric = 2 
  }
  
  files = list.files(path=dir, pattern = "^[results_:alnum:_:alnum:]")
  
  ma = matrix(unlist(strsplit(files, "_")),3)
  
  models_labels = unique(ma[2,])
  datasets_labels = unique(ma[3,])
  
  models = hash(models_labels, 1:length(models_labels))
  datasets = hash(datasets_labels, 1:length(datasets_labels))
  
  mdim = c(length(models)*n_metric, length(datasets), trials)
  f1 = array(0, mdim, dimnames = list(1:mdim[1],datasets_labels,1:trials))
  
  for (i in 1:length(files)) {
    file = paste(dir, files[i], sep = "/")

    results = result.load(file, trials)
    
    m = models[[ma[2,i]]]
    d = datasets[[ma[3,i]]]
    
    for (j in 1:n_metric) {
      f1[(m-1)*n_metric + j, d, ] = results[,j]
    }
  }

  return(list(f1, models_labels, datasets_labels))
}

stats.sigficant.winner <- function(measures, model_labels, means,
                                   p.adjust="bonf", conf.level = 0.95){
  if(conf.level > 1 || conf.level <= 0)
    stop("conf.level must be in (0,1]")
  
  if(length(model_labels) <= 1){
    return(array(F, 1))
  }
  
  library(Matrix)
  
  trials = length(measures)/length(model_labels)

  pv = pairwise.t.test(measures, rep(model_labels, each=trials), p.adjust=p.adjust)$p.value
  
  pv = pv > (1 - conf.level)
  #/length(model_labels)
  pv_dim = dim(pv)[1]
  pv = rbind(matrix(F, nrow=1, ncol=pv_dim), pv)
  pv = cbind(pv, matrix(F, nrow=pv_dim + 1, ncol=1))
  pv = forceSymmetric(t(pv))
  diag(pv) = T
  
  sorted <- order(model_labels)
  rownames(pv) <- colnames(pv) <- model_labels[sorted]
  
  return(pv)
}

stats.sigficant.winner.table <- function(measures, means, rownames, colnames,
                                         p.adjust = "bonf"){
  nrow = dim(measures)[1]
  ncol = dim(measures)[2]
  
  nmetric = nrow/length(rownames)
  div = 1:(nrow)%%nmetric
  
  winner_table = array(F, c(nrow, ncol))

  for(c in 1:ncol){
    for(i in 0:(nmetric-1)){
      idx = (div == i)

      tab <- stats.sigficant.winner(
        t(measures[idx,c,]), rownames, means[idx,c], p.adjust = p.adjust)
      winner_table[idx, c] <- tab[rownames, rownames[which.max(means[idx,c])]]
    }
  }
  
  return(winner_table)
}

print_meas <- function(measures, err, rownames, colnames, measnames, emphasize,
                       caption = "Table", label = "tab:tab", type="tex"){
  nrow = dim(measures)[1]
  ncol = dim(measures)[2]
  
  nmetric = nrow/length(rownames)
  div = 1:(nrow)%%nmetric
  
  df = matrix(paste(measures, err, sep = " $\\pm$  "),
              nrow = nrow, ncol = ncol, 
              dimnames = list(rep(rownames,nmetric), colnames))
  
  df[emphasize] <- paste("\\bf{",df[emphasize],"}", sep="")
  if(nmetric > 1){
    df <- cbind(matrix("", nrow = nrow, ncol = 2), df)
    
    for(i in 0:(nmetric-1)){
      idx = (div == i)
      if (i == 1){
        df[idx, 1] = paste("\\multirow{2}{*}{", rownames, "}",df[idx,1], sep="")
      }
      df[idx, 2] = measnames[nmetric-i]
    }
  
    sums = apply(emphasize, 1, sum)
    even = 1:(nrow/nmetric)*nmetric
    # sort using rank aggregation
    x <- t(apply(measures[even,], 2, order, decreasing=T))
    x <- rbind(t(apply(measures[even-1,], 2, order, decreasing=T)))
  
    rank <- RankAggreg(x, nrow/nmetric, verbose = F)
    
    df[even,] = df[even[rank$top.list], ]
    df[even-1,] = df[even[rank$top.list]-1, ]
    sums[even] = sums[even[rank$top.list]]
    sums[even-1] = sums[even[rank$top.list]-1]
    # sort by biggest winner
    o = even[order(sums[even]+sums[even-1], decreasing = T)]
    df[even,] = df[o, ]
    df[even-1,] = df[o-1, ]
  }
  ss <- seq(0, nrow, nmetric)
  tab <- xtable(df, caption = caption, label = label)
  if(nmetric > 1){
    print(tab, sanitize.text.function = identity,
          include.rownames = FALSE, hline.after = c(-1, 0),
          add.to.row = list(
            pos = as.list(ss),
            command = rep(paste("\\cline{3-", 2 + ncol, "}", sep=""), length(ss))))
  }else{
    print(tab, sanitize.text.function = identity)
  }
    
}

build_table <- function(dir_path, caption, label, trials=5){
  source("~/Documents/Master Degree/Master Project/Implementation/LazyNN_RF/experiments/reports/utils.R")
  # load results from directory
  # and extract information such as
  # metric (f1-measure), models and
  # datasets used
  results = result.load.dir(dir_path, trials)
  
  f1 = results[[1]]
  models_labels = toupper(results[[2]])
  datasets_labels = toupper(results[[3]])
  
  f1_avg = round(apply(f1, c(1,2), mean)*100, digits=2)
  f1_sd = round(apply(f1, c(1,2), sd)*100, digits=2)
  
  winner_table <- stats.sigficant.winner.table(f1, f1_avg, models_labels,
                                               datasets_labels, p.adjust = "bonf")
  print_meas(f1_avg, f1_sd, models_labels, datasets_labels,
             c("microF1", "macroF1"), winner_table, 
             caption = caption, label = label)
  
  winning_counts <- matrix(apply(winner_table, 1, sum), byrow = T, ncol = 2,
         dimnames = list(models_labels, c("MicroF1", "MacroF1")))
  return(winning_counts)
}