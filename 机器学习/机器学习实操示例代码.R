# ============================
# Iris 分类案例（tidymodels）
# - 三个模型：multinom (nnet), random forest (ranger), KNN (kknn)
# - RF 与 KNN 进行超参数调优（5 折交叉验证）
# - 模型评估（准确率、混淆矩阵、ROC），变量重要性（vip）
# ============================

# 如果缺包请先安装（取消注释运行一次）
# install.packages(c("tidymodels", "vip", "ranger"))

library(tidymodels)
library(vip)
library(ranger)

tidymodels_prefer()  # 解决可能的名称冲突
set.seed(123)

# ----------------------------
# 1. 数据加载与划分（80/20，按 Species 分层）
# ----------------------------
data(iris)
iris <- as.data.frame(iris)
iris_split <- initial_split(iris, prop = 0.8, strata = Species)
iris_train <- training(iris_split)
iris_test  <- testing(iris_split)

# ----------------------------
# 2. Recipe：标准化数值预测变量
# ----------------------------
iris_rec <- recipe(Species ~ ., data = iris_train) %>%
  step_normalize(all_numeric_predictors())

# ----------------------------
# 3. 模型规格（parsnip）
# ----------------------------
# 3.1 多类逻辑回归（nnet 引擎）
log_mod <- multinom_reg(mode = "classification") %>%
  set_engine("nnet")

# 3.2 随机森林（ranger 引擎），mtry/min_n 需要调优；开启 importance="permutation"
rf_mod <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

# 3.3 KNN（kknn 引擎），neighbors 需要调优
knn_mod <- nearest_neighbor(neighbors = tune(), weight_func = "rectangular", dist_power = 2) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# ----------------------------
# 4. Workflow：把 recipe 和各模型绑定
# ----------------------------
log_wf <- workflow() %>% add_model(log_mod) %>% add_recipe(iris_rec)
rf_wf  <- workflow() %>% add_model(rf_mod)  %>% add_recipe(iris_rec)
knn_wf <- workflow() %>% add_model(knn_mod) %>% add_recipe(iris_rec)

# ----------------------------
# 5. 交叉验证折（5 折，分层）
# ----------------------------
set.seed(2025)
iris_folds <- vfold_cv(iris_train, v = 5, strata = Species)

# 评价指标
metrics <- metric_set(accuracy, roc_auc)

# tune 控制项：保存每次预测
ctrl <- control_grid(save_pred = TRUE)

# ----------------------------
# 6. 随机森林调参（提取参数并 finalize）
# ----------------------------
# 提取可调参数集合（从 workflow）
rf_params <- rf_wf %>% extract_parameter_set_dials()
# finalize 需要传入 predictors（去掉响应列）
predictors_df <- iris_train %>% select(-Species)
rf_params_final <- dials::finalize(rf_params, x = predictors_df)

# 随机搜索网格
set.seed(234)
rf_grid <- dials::grid_random(rf_params_final, size = 20)

# 调参
set.seed(234)
rf_res <- tune_grid(
  rf_wf,
  resamples = iris_folds,
  grid = rf_grid,
  metrics = metrics,
  control = ctrl
)

# 查看最好结果（按准确率）
show_best(rf_res, metric = "accuracy")
best_rf <- select_best(rf_res, metric = "accuracy")
print(best_rf)

# ----------------------------
# 7. KNN 调参（同样提取并 finalize）
# ----------------------------
knn_params <- knn_wf %>% extract_parameter_set_dials()
knn_params_final <- dials::finalize(knn_params, x = predictors_df)

# 使用 grid_regular 生成 neighbors 的网格（或随机）
knn_grid <- dials::grid_regular(knn_params_final, levels = 10)

set.seed(235)
knn_res <- tune_grid(
  knn_wf,
  resamples = iris_folds,
  grid = knn_grid,
  metrics = metrics,
  control = ctrl
)

show_best(knn_res, metric = "accuracy")
best_knn <- select_best(knn_res, metric = "accuracy")
print(best_knn)

# ----------------------------
# 8. 训练最终模型（使用训练集）
#    - 逻辑回归（无调参）
#    - 随机森林 & KNN 使用选好的最佳参数
# ----------------------------
# 8.1 逻辑回归最终模型
# 逻辑回归最终模型（无需 finalize_workflow）
log_fit <- fit(log_wf, data = iris_train)

# 8.2 随机森林最终模型
final_rf_wf <- finalize_workflow(rf_wf, best_rf)
rf_fit <- fit(final_rf_wf, data = iris_train)

# 8.3 KNN 最终模型
final_knn_wf <- finalize_workflow(knn_wf, best_knn)
knn_fit <- fit(final_knn_wf, data = iris_train)

# ----------------------------
# 9. 在测试集上预测与评估（类别 + 概率）
# ----------------------------
# 函数：统一计算并返回评估结果（混淆矩阵、准确率、ROC AUC）
eval_model <- function(fit, test_data, model_name = "model") {
  pred_class <- predict(fit, test_data)
  pred_prob  <- predict(fit, test_data, type = "prob")
  res <- bind_cols(test_data, pred_class, pred_prob)
  cat("===== Model:", model_name, "=====\n")
  print(conf_mat(res, truth = Species, estimate = .pred_class))
  print(accuracy(res, truth = Species, estimate = .pred_class))
  # 多类 ROC AUC（需提供每一类的 .pred_* 列）
  auc_res <- tryCatch({
    roc_auc(res, truth = Species, .pred_setosa, .pred_versicolor, .pred_virginica, estimator = "macro_weighted")
  }, error = function(e) NA)
  print(auc_res)
  return(res)
}

res_log <- eval_model(log_fit, iris_test, "Multinomial (nnet)")
res_rf  <- eval_model(rf_fit, iris_test, "Random Forest")
res_knn <- eval_model(knn_fit, iris_test, "KNN")

# ----------------------------
# 10. 绘制每个模型的 ROC 曲线（以 setosa 为例）并比较
# ----------------------------
# 多分类 ROC 曲线（Hand-Till 方法）
roc_res_rf <- roc_curve(
  res_rf,
  truth = Species,
  .pred_setosa,
  .pred_versicolor,
  .pred_virginica,
  estimator = "hand_till"
)

roc_res_knn <- roc_curve(
  res_knn,
  truth = Species,
  .pred_setosa,
  .pred_versicolor,
  .pred_virginica,
  estimator = "hand_till"
)

roc_res_log <- roc_curve(
  res_log,
  truth = Species,
  .pred_setosa,
  .pred_versicolor,
  .pred_virginica,
  estimator = "hand_till"
)

library(ggplot2)

autoplot(roc_res_rf) + ggtitle("Random Forest ROC (multi-class)")
autoplot(roc_res_knn) + ggtitle("KNN ROC (multi-class)")
autoplot(roc_res_log) + ggtitle("Multinomial ROC (nnet)")


# ----------------------------
# 11. 变量重要性（针对随机森林）
# ----------------------------
library(vip)

# 提取 parsnip 底层 ranger 对象
rf_parsnip <- extract_fit_parsnip(rf_fit)
ranger_obj <- rf_parsnip$fit

# ----------------------------
# VIP 绘图（基于训练时 ranger importance）
# ----------------------------
vip(ranger_obj) + ggtitle("Random Forest - Variable Importance (vip)")


# ============================
# End of script
# ============================
