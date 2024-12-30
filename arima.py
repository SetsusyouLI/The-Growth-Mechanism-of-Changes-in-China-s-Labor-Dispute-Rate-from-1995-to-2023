# 导入必要的库
import pandas as pd
import numpy as np
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
import itertools

# 忽略ARIMA模型拟合时的警告
warnings.filterwarnings("ignore")

# 1. 数据加载与预处理

# 数据存放路径
data_path = r'C:\Users\81286\Desktop\p.xlsx'

# 检查文件是否存在
if not os.path.exists(data_path):
    raise FileNotFoundError(f"数据文件未找到，请检查路径: {data_path}")

# 读取Excel文件
df = pd.read_excel(data_path)

# 查看前几行数据
print("数据预览:")
print(df.head())

# 检查缺失值
missing_values = df.isnull().sum()
print("\n缺失值统计:")
print(missing_values)

# 处理缺失值
# 这里选择删除含有缺失值的行，您也可以选择填补缺失值
df = df.dropna()
print("\n删除缺失值后的数据维度:", df.shape)

# 确认 'L.Dispute%' 列存在
if 'L.Dispute%' not in df.columns:
    raise ValueError("数据中不包含 'L.Dispute%' 列。")

# 提取 'L.Dispute%' 列作为时间序列数据
y = df['L.Dispute%']

# 2. 模型拟合

# 定义p和q的范围
p_values = range(0, 6)
q_values = range(0, 6)

# 创建所有可能的p和q组合
pq_combinations = list(itertools.product(p_values, q_values))

# 初始化一个列表来存储模型结果
model_results = []

# 遍历所有p和q的组合
for p, q in pq_combinations:
    try:
        # 定义并拟合ARIMA模型（p,1,q）
        model = ARIMA(y, order=(p, 1, q))
        model_fit = model.fit()

        # 获取AIC和BIC
        aic = model_fit.aic
        bic = model_fit.bic

        # 将结果添加到列表中
        model_results.append({'p': p, 'q': q, 'AIC': aic, 'BIC': bic})

        print(f"Fitted ARIMA({p},1,{q}) - AIC: {aic:.2f}, BIC: {bic:.2f}")

    except Exception as e:
        # 如果模型拟合失败，打印错误信息并继续
        print(f"Failed to fit ARIMA({p},1,{q}): {e}")

# 3. 结果汇总与最佳模型选择

# 检查是否有成功拟合的模型
if not model_results:
    print("\n没有成功拟合任何ARIMA模型。")
else:
    # 创建DataFrame
    results_df = pd.DataFrame(model_results)

    # 根据AIC和BIC排序
    model_results_sorted_aic = results_df.sort_values(by='AIC').reset_index(drop=True)
    model_results_sorted_bic = results_df.sort_values(by='BIC').reset_index(drop=True)

    print("\n所有成功拟合的ARIMA（p,1,q）模型及其AIC和BIC值:")
    print(model_results_sorted_aic)

    # 确定最佳模型（AIC最小）
    best_model_aic = model_results_sorted_aic.iloc[0]
    best_p_aic, best_q_aic, best_aic, best_bic_aic = best_model_aic

    # 确定最佳模型（BIC最小）
    best_model_bic = model_results_sorted_bic.iloc[0]
    best_p_bic, best_q_bic, best_aic_bic, best_bic_bic = best_model_bic

    print("\n最佳模型（基于AIC）:")
    print(f"ARIMA({int(best_p_aic)},1,{int(best_q_aic)}) - AIC: {best_aic:.2f}, BIC: {best_bic_aic:.2f}")

    print("\n最佳模型（基于BIC）:")
    print(f"ARIMA({int(best_p_bic)},1,{int(best_q_bic)}) - AIC: {best_aic_bic:.2f}, BIC: {best_bic_bic:.2f}")

    # 如果基于AIC和BIC的最佳模型相同，则只打印一次
    if (best_p_aic, best_q_aic) == (best_p_bic, best_q_bic):
        print("\n最终最佳模型（基于AIC和BIC）:")
        print(f"ARIMA({int(best_p_aic)},1,{int(best_q_aic)}) - AIC: {best_aic:.2f}, BIC: {best_bic_aic:.2f}")
    else:
        print("\n基于AIC和BIC的最佳模型不同。")
        print("您可以根据具体需求选择其中一个作为最终模型。")

    # 可选：保存模型结果到Excel
    output_path = r'C:\Users\81286\Desktop\arimaresults.xlsx'
    results_df_sorted_aic = model_results_sorted_aic.copy()
    results_df_sorted_aic.to_excel(output_path, index=False)
    print(f"\n所有模型的结果已保存到: {output_path}")

    # 4. 拟合最佳模型并打印摘要
    # 选择基于AIC的最佳模型
    print("\n拟合最佳模型（基于AIC）并打印摘要:")
    try:
        best_p, best_q = int(best_p_aic), int(best_q_aic)
        best_arima_model = ARIMA(y, order=(best_p, 1, best_q)).fit()
        print(best_arima_model.summary())
    except Exception as e:
        print(f"无法拟合最佳模型ARIMA({best_p_aic},1,{best_q_aic}): {e}")

    # 4. 拟合最佳模型并打印摘要
    # 选择基于AIC的最佳模型
    print("\n拟合最佳模型（基于BIC）并打印摘要:")
    try:
        best_p, best_q = int(best_p_bic), int(best_q_bic)
        best_arima_model = ARIMA(y, order=(best_p, 1, best_q)).fit()
        print(best_arima_model.summary())
    except Exception as e:
        print(f"无法拟合最佳模型ARIMA({best_p_bic},1,{best_q_bic}): {e}")
