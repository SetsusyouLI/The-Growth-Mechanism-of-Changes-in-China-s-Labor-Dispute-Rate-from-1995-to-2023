import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV, Ridge
import statsmodels.api as sm
from scipy.stats import f
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

def ridge_r_squared(y_true, y_pred, y_mean, lambda_, beta):
    """
    计算岭回归的修正R²
    """
    # 计算惩罚项（不包括截距）
    penalty = lambda_ * np.sum(beta[1:]**2)
    
    # 计算修正的残差平方和
    rss_adj = np.sum((y_true - y_pred)**2) + penalty
    
    # 计算总平方和
    tss = np.sum((y_true - y_mean)**2)
    
    # 计算修正的R²
    r_squared_adj = 1 - rss_adj/tss
    
    return r_squared_adj

def train_arimax_ridge(y, X, lag, alphas=np.logspace(-3, 3, 100)):
    """
    使用岭回归（自动选择最佳alpha），并计算相关统计量。
    
    参数:
    y: 因变量 (Pandas Series)
    X: 自变量 (Pandas DataFrame)
    lag: 滞后阶数 (整数)
    alphas: 岭回归的候选正则化参数列表 (array-like)
    
    返回:
    result: 包含模型结果的字典
    """
    # 创建滞后自变量
    X_lagged = X.shift(lag)
    
    # 合并滞后后的自变量和因变量，并删除含有 NaN 的行
    data = pd.concat([y, X_lagged], axis=1).dropna()
    y_train = data['L.Dispute%']
    X_train = data.drop('L.Dispute%', axis=1)
    
    # 确保所有自变量都是数值类型
    try:
        X_train = X_train.apply(pd.to_numeric, errors='raise')
    except ValueError as e:
        print(f"数据转换错误: {e}")
        print("请检查自变量是否包含非数值数据。")
        raise
    
    # 添加常数项
    X_train = sm.add_constant(X_train)
    
    # 转换为矩阵，确保是 float 类型
    X_matrix = X_train.values.astype(float)
    y_matrix = y_train.values.astype(float)
    
    # 使用 RidgeCV 选择最佳 alpha，移除 store_cv_values 以消除警告
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=False)
    ridge_cv.fit(X_matrix, y_matrix)
    best_alpha = ridge_cv.alpha_
    
    # 使用最佳 alpha 拟合 Ridge 回归
    ridge = Ridge(alpha=best_alpha, fit_intercept=False)
    ridge.fit(X_matrix, y_matrix)
    coef = ridge.coef_
    
    # 预测
    y_pred = ridge.predict(X_matrix)
    
    # 计算修正的R²
    y_mean = np.mean(y_matrix)
    r_squared_modified = ridge_r_squared(y_matrix, y_pred, y_mean, best_alpha, coef)
    
    # 计算残差
    residuals = y_matrix - y_pred
    
    # 计算残差方差
    sigma_squared = np.var(residuals, ddof=X_matrix.shape[1])
    
    # 计算 (X^T X + lambda * I)^(-1)
    XTX = X_matrix.T @ X_matrix
    lambda_I = best_alpha * np.eye(X_matrix.shape[1])
    try:
        inv_XTX_lambdaI = np.linalg.inv(XTX + lambda_I)
    except np.linalg.LinAlgError as e:
        print(f"矩阵不可逆错误: {e}")
        raise
    
    # 计算标准误差
    se = np.sqrt(np.diag(inv_XTX_lambdaI) * sigma_squared)
    # 计算 t 值
    t_values = coef / se
    
    # 自由度
    n = X_matrix.shape[0]
    k = X_matrix.shape[1] - 1  # 不包括常数项
    df_model = k
    df_resid = n - k - 1
    
    # 计算 AIC 和 BIC
    ss_total = np.sum((y_matrix - np.mean(y_matrix))**2)
    ss_res = np.sum(residuals**2)
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + np.log(n) * k
    
    # Ljung-Box 检验
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    Q_stat = lb_test['lb_stat'].values[0]    # 添加Q统计量
    prob_Q = lb_test['lb_pvalue'].values[0]
    
    # Jarque-Bera 检验
    jb_test = jarque_bera(residuals)
    jb_stat = jb_test[0]
    prob_JB = jb_test[1]
    
    # Breusch-Pagan 异方差性检验
    bp_test = het_breuschpagan(residuals, X_train)
    H = bp_test[0]        # Breusch-Pagan 测试统计量
    Prob_H = bp_test[1]   # Breusch-Pagan p 值
    
    # 计算 F 统计量
    ss_reg = ss_total - ss_res  # 回归平方和
    f_stat = (ss_reg / df_model) / (ss_res / df_resid)
    f_pvalue = 1 - f.cdf(f_stat, df_model, df_resid)
    
    # 构建结果字典，添加 alpha、F 统计量和 p 值
    result = {
        'Lag': lag,
        'Best_alpha': best_alpha,
        'R_squared': r_squared_modified,  # 使用修正后的R²
        'sigma2': sigma_squared,
        'AIC': aic,
        'BIC': bic,
        'Q_stat': Q_stat,
        'Prob(Q)': prob_Q,
        'H': H,
        'Prob(H)': Prob_H,
        'JB': jb_stat,
        'Prob(JB)': prob_JB,
        'F_stat': f_stat,
        'F_pvalue': f_pvalue,
        'coefficients': coef
    }
    
    return result

# 读取数据
file_path = r'C:\Users\81286\Desktop\biao1.xlsx'
df = pd.read_excel(file_path)

# 确保 'year' 列为整数
df['year'] = df['year'].astype(int)

# 确保数据按"year"排序
df = df.sort_values('year')

# 创建自变量的平方项
df['GDP.P.C_sq'] = df['GDP.P.C'] ** 2

# 定义因变量
y = df['L.Dispute%']

# 定义原有的自变量
X = df[['L.Dispute-1', 'GDP.P.C', 'Education', 'LawyerS.', 'FDI%', 
        'R.14', 'Gini', 'Urbanization%', 'L-Law', 'L.Contrac-Law', 'L.Arbitration-Law', 
            'S.Insurance-Law', 'J.Inter.']]

# --------------------------
# 添加关键年份虚拟变量（断点年份：1996、1997、2008）
# --------------------------
df['year_1996'] = (df['year'] == 1996).astype(int)
df['year_1997'] = (df['year'] == 1997).astype(int)
df['year_2008'] = (df['year'] == 2008).astype(int)

# 将关键年份虚拟变量拼接到自变量中
X = pd.concat([X, df[['year_1996', 'year_1997', 'year_2008']]], axis=1)

# 确认自变量数据类型
print("自变量 X 的数据类型：")
print(X.dtypes)

# 尝试将所有列转换为数值类型
try:
    X = X.apply(pd.to_numeric, errors='raise')
except ValueError as e:
    print(f"数据转换错误: {e}")
    print("请检查自变量是否包含非数值数据。")
    raise

# 检查是否有 NaN 值
if X.isnull().values.any():
    print("\n存在 NaN 值，进行删除。")
    X = X.dropna()
    y = y.loc[X.index]

# 定义滞后阶数
lags = [0, 1, 2, 3, 4, 5]

# 存储结果
results = []

# 循环训练模型
for lag in lags:
    try:
        res = train_arimax_ridge(y, X, lag)  # 使用默认的 alphas
        results.append(res)
    except Exception as e:
        print(f"在滞后阶数 {lag} 时发生错误: {e}")

# 将结果整理为 DataFrame
output = []
for res in results:
    coef = res['coefficients']
    coef_names = ['const'] + list(X.columns)
    record = {
        'Lag': res['Lag'],
        'Best_alpha': res['Best_alpha'],
        'R_squared': res['R_squared'],
        'sigma2': res['sigma2'],
        'AIC': res['AIC'],
        'BIC': res['BIC'],
        'Ljung-Box': res['Q_stat'],
        'Prob(Q)': res['Prob(Q)'],
        'Heteroskedasticity': res['H'],
        'Prob(H)': res['Prob(H)'],
        'Jarque-Bera': res['JB'],
        'Prob(JB)': res['Prob(JB)'],
        'F_stat': res['F_stat'],
        'F_pvalue': res['F_pvalue']
    }
    for i, name in enumerate(coef_names):
        record[f'coef_{name}'] = coef[i]
    output.append(record)

output_df = pd.DataFrame(output)

# 保存到桌面
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
save_path = os.path.join(desktop_path, '岭回归全部（r修正）.xlsx')
output_df.to_excel(save_path, index=False)

print(f"结果已保存到 {save_path}")
