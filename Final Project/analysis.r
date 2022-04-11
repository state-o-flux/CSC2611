library(openxlsx)
library(tidyr)
library(ggplot2)

setwd('C:/Users/Liam/Documents/PhD/CSC2611/Final Project/Analysis')

gen_log_model <- function(model, df, grade=NA, printer=FALSE) {
  if(!is.na(grade)) {
    df <- df[df$grade == grade,]
  }
  log_model <- if(model == "t1") {
    glm(learned ~ t1_LP, data = df, family = "binomial")
  } else if(model == "t2") {
    glm(learned ~ t2_LP, data = df, family = "binomial")
  } else if(model == "t3") {
    glm(learned ~ t3_LP, data = df, family = "binomial")
  } else if(model == "baseline") {
    glm(learned ~ baseline_LP, data = df, family = "binomial")
  }
  
  if(printer) {
    coefs <- round(coef(summary(log_model))[2,c(1,4)],4)
    print(coefs)
  }
  return(log_model)
}

plotter_overall <- function(df, models) {
  newdf <- lapply(models, function(x) {
    newdf_1 <- data.frame(LP = rep(seq(-3, 3, length.out = 50)))
    names(newdf_1) <- names(x$coefficients)[2]
    
    newdf_2 <- cbind(newdf_1, predict(x, newdata = newdf_1, type = "link", se = TRUE))
    newdf_2 <- within(newdf_2, {
      PredictedProb <- plogis(fit)
      LL <- plogis(fit - (1.96 * se.fit))
      UL <- plogis(fit + (1.96 * se.fit))
    })
    names(newdf_2)[1] <- "LP"
    newdf_2
  })
  newdf <- do.call(rbind, newdf)
  theory <- c(rep("Preferential Attachment", 50), rep("Preferential Acquisition", 50), rep("Lure of Associates", 50), rep("Frequency Baseline", 50))
  newdf <- cbind(theory, newdf)
  newdf$theory <- factor(newdf$theory, levels = c("Preferential Attachment", "Preferential Acquisition", "Lure of Associates", "Frequency Baseline"))
  
  ggplot(newdf, aes(x = LP, y = PredictedProb)) +
    geom_ribbon(aes(ymin = LL, ymax = UL), alpha = 0.2) +
    geom_line(size = 1) +
    labs(x = "Learn Potential (z-score)", y = "Probability that word is Learned") +
    scale_y_continuous(limits = c(0,1)) +
    facet_wrap(~theory, nrow=1) +
    theme_bw()
}

plotter_bygrade <- function(df, model, theory) {
  
  grades <- unique(df$grade)

  newdf_1 <- data.frame(grade = sort(rep(grades, 50)), LP = rep(seq(-3, 3, length.out = 50),length(grades)))
  names(newdf_1)[2] <- names(model$coefficients)[2]
  newdf_2 <- cbind(newdf_1, predict(model, newdata = newdf_1, type = "link", se = TRUE))
  
  newdf_2 <- within(newdf_2, {
    PredictedProb <- plogis(fit)
    LL <- plogis(fit - (1.96 * se.fit))
    UL <- plogis(fit + (1.96 * se.fit))
  })
  names(newdf_2)[2] <- "LP"
  
  grade_names <- as.list(paste("Grade", grades, sep = " "))
  grade_labeller <- function(variable,value){
    return(grade_names[value])
  }
  
  ggplot(newdf_2, aes(x = LP, y = PredictedProb)) +
    geom_ribbon(aes(ymin = LL, ymax = UL), alpha = 0.2) +
    geom_line(size = 1) +
    labs(x = "Learn Potential (z-score)", y = "Probability that word is Learned", title = theory) +
    facet_wrap(~grade, labeller = as_labeller(grade_labeller), nrow = 1) +
    theme_bw()
}

tabler_overall <- function(models, names) {
  result <- lapply(models, function(x) round(coef(summary(x))[2,],3))
  result <- data.frame(do.call(rbind, result))
  result <- cbind(names, result)
  names(result)[1] <- "Theory"
  names(result)[5] <- "p"
  return(result)
}

test_theories <- function(window, by_grade = F) {
  df <- paste0('learningPotential_', window, '.xlsx')
  df <- read.xlsx(df)
  df_long <- pivot_longer(df, c(t1_LP, t2_LP, t3_LP, baseline_LP), names_to = "theory")
  grades <- unique(df$grade)
  
  t1_model <- gen_log_model("t1", df)
  t2_model <- gen_log_model("t2", df)
  t3_model <- gen_log_model("t3", df)
  baseline_model <- gen_log_model("baseline", df)
  
  models <- list(t1_model, t2_model, t3_model, baseline_model)
  names <- c("Preferential Attachment", "Preferential Acquisition", "Lure of Associates", "Frequency Baseline")
  
  print(tabler_overall(models, names))
  print(plotter_overall(df_long, models))
  
  if(by_grade) {
    t1_grade_models <- lapply(grades, function(x) gen_log_model('t1', df, x))
    t1_grade_model <- glm(learned ~ t1_LP * grade, data = df, family = "binomial")
    print(plotter_bygrade(df, t1_grade_model, names[1]))

    t2_grade_models <- lapply(grades, function(x) gen_log_model('t2', df, x))
    t2_grade_model <- glm(learned ~ t2_LP * grade, data = df, family = "binomial")
    print(plotter_bygrade(df, t2_grade_model, names[2]))
    
    t3_grade_models <- lapply(grades, function(x) gen_log_model('t3', df, x))
    t3_grade_model <- glm(learned ~ t3_LP * grade, data = df, family = "binomial")
    print(plotter_bygrade(df, t3_grade_model, names[3]))
    
    baseline_grade_models <- lapply(grades, function(x) gen_log_model('baseline', df, x))
    baseline_grade_model <- glm(learned ~ baseline_LP * grade, data = df, family = "binomial")
    print(plotter_bygrade(df, baseline_grade_model, names[4]))
  }
}

test_theories(window = "2", by_grade = T)
test_theories(window = "3", by_grade = T)
test_theories(window = "4", by_grade = T)
