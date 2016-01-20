require(hash)
require(caret)

createConfusionMatrix <- function(act, pred) {
  # You've mentioned that neither actual nor predicted may give a complete
  # picture of the available classes, hence:
  numClasses <- max(act,pred)
  #  max(length(unique(act)), length(unique(pred)))
  # Sort predicted and actual as it simplifies what's next. You can make this
  # faster by storing `order(act)` in a temporary variable.
  pred <- pred[order(act)]
  act  <- act[order(act)]
  sapply(split(pred, act), tabulate, nbins=numClasses)
}


f1_measure <- function(y, pred, type="both"){
  cm = as.matrix(confusionMatrix(table(factor(pred,levels=min(y):max(y)),factor(y,levels=min(y):max(y)))))
  
  tp = diag(cm)
  fp = apply(cm, 2, sum) - tp
  fn = apply(cm, 1, sum) - tp
  
  fp[fp==0] = 1
  fn[fn==0] = 1
  
  pres = tp/(tp+fp) 
  recall = tp/(tp+fn) 
  
  # computing macro f1  
  mac_p = mean(tp/(tp+fp))  
  mac_r = mean(tp/(tp+fn))  
  mac_f1 = 2*(mac_p*mac_r)/(mac_p+mac_r)
  
  # computing micro f1
  mic_p = sum(tp)/(sum(tp)+sum(fp))
  mic_r = sum(tp)/(sum(tp)+sum(fn))
  mic_f1 = 2*(mic_p*mic_r)/(mic_p+mic_r)  
  if(type == "both"){
    return(list(mic_f1,mac_f1))
  }else if(type == "macro"){
    return(mac_f1)
  }else{
    return(mic_f1)
  }
}