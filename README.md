# 企业岗位定价模型（JobPricingProject ）

**系统流程：企业分层 → 曝光预测 → 定价预测**
![app](.\doc\app.png)

![app1](./doc\app1.png)

------

##  功能概述

- **端到端定价链路**
  - **企业分层（Enterprise Tier）**：基于企业规模、品牌等级、融资阶段及人才关怀指数等结构化特征，生成企业档位评估结果
  - **曝光预测（Impressions）**：采用 XGBoost 回归模型预测岗位曝光量，并在 log1p 域进行训练与评估
  - **定价推荐（Price）**：采用 MLP 回归模型输出岗位推荐价格，支持 log 域训练与推理还原，提升长尾稳健性
- **岗位健康度特征**
  - 支持将 `job_health_score` 作为定价特征注入，提供三种模型版本：
    - `none`：不引入健康度特征（Baseline）
    - `lstm`：基于 LSTM 生成健康度分数
    - `transformer`：基于 Transformer 生成健康度分数
- **离线实验与可视化评估**
  - 自动生成 `metrics_summary.csv` / `impressions_metrics.csv` 等评估汇总文件
  - 输出 True vs Pred 散点图、误差直方图、模型对比图等可视化结果，便于对不同模型版本进行横向对比与误差分析
- **前端演示（Streamlit）**
  - 支持企业与岗位信息交互式输入，一键输出企业档位、预测曝光、推荐价格及关键指标拆解结果，用于演示与验收

------

##  项目目录结构

```
JobPricingProject/
├── data/
│   ├── raw/                    
│   └── processed/              
├── models/                     
├── reports/                   
├── src/
│   ├── config.py              
│   ├── utils.py             
│   ├── preprocess.py           
│   ├── train.py              
│   ├── predict.py              
│   └── experiment.py       
├── app.py                  
└── requirements.txt
```

------

##  环境与依赖

建议 Python 3.7+（推荐 3.9）

### 安装依赖

```bash
pip install -r requirements.txt
```

------

##  快速开始

> 在项目根目录执行（例如：`E:\JobPricingProject`）

### 1) 数据预处理

```bash
python -m src.preprocess
```

------

### 2) 训练模型

#### 2.1 None

```bash
python -m src.train --health_model none
```

#### 2.2 LSTM

```bash
python -m src.train --health_model lstm
```

#### 2.3 Transformer

```bash
python -m src.train --health_model transformer
```

------

### 3) 离线实验评估

```bash
python -m src.experiment
```

------

### 4) 启动 Streamlit 前端

```bash
streamlit run app.py
```

------

##  实验结果

### Impressions: True vs Pred

> ![impressions_true_vs_pred](.\reports\impressions_true_vs_pred.png)

### Pricing Model Comparison

> ![model_compare_mae_rmse](.\reports\model_compare_mae_rmse.png)

### Price Error Histogram（none / lstm / transformer）

> ![price_calibration_decile_none](.\reports\price_calibration_decile_none.png)
>
> ![price_calibration_decile_lstm](.\reports\price_calibration_decile_lstm.png)
>
> ![price_calibration_decile_transformer](.\reports\price_calibration_decile_transformer.png)

### Price: True vs Pred（none / lstm / transformer）

> ![price_error_hist_none](.\reports\price_error_hist_none.png)
>
> ![price_error_hist_lstm](.\reports\price_error_hist_lstm.png)
>
> ![price_error_hist_transformer](.\reports\price_error_hist_transformer.png)

------

## 指标说明

### Impressions（曝光预测）

- **评估域**：采用 **log1p 域**评估
- **指标**：`MAE(log1p)`、`RMSE(log1p)`

### Price（定价预测）

- **评估域**：同时在两种域评估
  - **原始价格域**：`MAE`、`RMSE`
  - **log1p 域**：`MAE(log1p)`、`RMSE(log1p)`
- **分桶评估**：按价格区间/分位分桶分别计算误差，用于反映不同价格段的预测质量。

------

##  常见问题（FAQ）

------

##  License

MIT 

------

##  Acknowledgements

- XGBoost / PyTorch / Streamlit / scikit-learn 社区生态支持