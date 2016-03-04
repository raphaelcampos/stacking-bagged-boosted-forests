require(hash)
require(caret)

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

f1_measure <- function(y, pred, type="both"){
  u = union(pred, y)
  t = table(factor(pred, u), factor(y, u))

  cm <- as.matrix(confusionMatrix(t))
  
  tp = diag(cm)
  fp = apply(cm, 2, sum) - tp
  fn = apply(cm, 1, sum) - tp
  
  # computing macro f1  
  mac_p = mean(prf_divide(tp, tp + fp))
  mac_r = mean(prf_divide(tp, tp + fn))
  mac_f1 = 2*(mac_p * mac_r)/(mac_p + mac_r)
  
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