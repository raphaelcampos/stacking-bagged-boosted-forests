library(xtable)

source("~/Documents/Master Degree/Master Project/Implementation/LazyNN_RF/experiments/reports/f1-measure.R")

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
  splits<- grep("#", lines)
  
  table = read.table(text = lines, header = F, fill = T, col.names = paste0("V", n_cols))
    
  y = parser_class_column(table$V2)
  pred = parser_class_column(table$V3)
  
  y <- split(y, splits)
  pred <- split(pred, splits)
  
  trials <- length(pred)
  
  #y = matrix(y, ncol = trials)
  #pred = matrix(pred, ncol = trials)
  
  results = array(0, c(trials, n_metric))
  
  for (j in 1:trials) {
    # Compute metric
    if (metric == "f1") {
      f1_meas = f1_measure(y[[j]] + 1, pred[[j]] + 1)
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
  
  pv = pairwise.t.test(measures, rep(model_labels, each=trials), 
                       paired = F, p.adjust=p.adjust)$p.value
 
  pv = pv > (1 - conf.level)
  pv_dim = dim(pv)[1]
  pv = rbind(matrix(F, nrow=1, ncol=pv_dim), pv)
  pv = cbind(pv, matrix(F, nrow=pv_dim + 1, ncol=1))
  pv = forceSymmetric(t(pv))
  diag(pv) = T
  
  sorted <- order(model_labels)
  rownames(pv) <- colnames(pv) <- model_labels[sorted]
  return(pv[model_labels, model_labels[which.max(means)]])
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
      winner_table[idx, c] <- stats.sigficant.winner(
        t(measures[idx,c,]), rownames, means[idx,c], p.adjust = p.adjust)
    }
  }
  
  return(winner_table)
}

print_meas <- function(measures, err, rownames, colnames, measnames, emphasize,
                       caption = "Table", type="tex"){
  nrow = dim(measures)[1]
  ncol = dim(measures)[2]
  
  nmetric = nrow/length(rownames)
  div = 1:(nrow)%%nmetric
  
  df = matrix(paste(measures, err, sep = " $\\pm$  "),
              nrow = nrow, ncol = ncol, 
              dimnames = list(1:nrow, colnames))
  
  df[emphasize] <- paste("\\bf{",df[emphasize],"}", sep="")
  
  df <- cbind(matrix("", nrow = nrow, ncol = 2), df)
  
  for(i in 0:(nmetric-1)){
    idx = (div == i)
    if (i == 1){
      df[idx, 1] = paste("\\multirow{2}{*}{", rownames, "}",df[idx,1], sep="")
    }
    df[idx, 2] = measnames[nmetric-i]
  }
  
  tab <- xtable(df, caption = caption)
  print(tab, sanitize.text.function = identity, include.rownames = FALSE)
}