require(hash)
require(caret)
require(PerfMeas)

# Create a confusion matrix from the given outcomes, whose rows correspond
# to the actual and the columns to the predicated classes.
createConfusionMatrix <- function(act, pred) {
  # You've mentioned that neither actual nor predicted may give a complete
  # picture of the available classes, hence:
  numClasses <- max(act, pred)
  # Sort predicted and actual as it simplifies what's next. You can make this
  # faster by storing `order(act)` in a temporary variable.
  pred <- pred[order(act)]
  act  <- act[order(act)]
  sapply(split(pred, act), tabulate, nbins=numClasses)
}

prf_divide <- function(numerator, denominator){
  result = numerator / denominator
  mask = denominator == 0.0
  
  # remove infs
  result[mask] = 0.0
  
  return(result)
}

binary_matrix <- function(y, ncol){
  id <- cbind(rowid=1:length(y), colid=y)
  bm <- matrix(0, nrow = length(y), ncol=ncol)
  bm[id] <- 1
  
  colnames(bm) <- 1:ncol;
  return(bm)
}

f1_measure <- function(y, pred, type="both"){
  u = union(pred, y)
  inc = min(u)*(-1) + 1
  u = u + inc
  pred = pred + inc
  y = y + inc
  
  t = table(factor(pred, u), factor(y , u))

  cm <- as.matrix(confusionMatrix(t))
  
  tp = diag(cm)
  fp = apply(cm, 2, sum) - tp
  fn = apply(cm, 1, sum) - tp
  
  # computing macro f1  
  Targets <- binary_matrix(y, max(u))
  Pred <- binary_matrix(pred, max(u))
  mac_f1 <- F.measure.single.over.classes(Targets[,u], Pred[,u])$average['F']
  
  # computing micro f1
  mic_p = prf_divide(sum(tp), sum(tp)+sum(fp))
  mic_r = prf_divide(sum(tp), sum(tp)+sum(fn))
  mic_f1 = 2*(mic_p*mic_r)/(mic_p+mic_r)  
  
  if(type == "both"){
    return(list(mic_f1,mac_f1))
  }else if(type == "macro"){
    return(mac_f1)
  }else{
    return(mic_f1)
  }
}