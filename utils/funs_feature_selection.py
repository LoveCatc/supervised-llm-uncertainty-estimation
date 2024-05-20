from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import r_regression

def lasso_select_k_features(X, y, k, alpha=0.001, features_already_selected = [],features_from_saved=False):
    if features_from_saved:
        return []
    k = min(k,X.shape[1]-len(features_already_selected))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    #alpha_range = np.logspace(-3, 2, 100, base=10)   # NOTE: need check here
    #lasso = LassoCV(alphas=alpha_range, cv=5, random_state=0, n_jobs=-1, verbose=1)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    print(np.sum(lasso.coef_>0))
    
    coefficients = np.abs(lasso.coef_)

    coefficients[features_already_selected] = 0

    ranked_features = np.argsort(coefficients)[::-1]
    selected_features = ranked_features[:k]

    return selected_features

def mutual_info_select_k_features(X,y,k,features_already_selected = [],features_from_saved=False):
    if features_from_saved:
        return []
    k = min(k,X.shape[1]-len(features_already_selected))
    mutual_features = mutual_info_regression(X,y)
    mutual_features = np.abs(np.array(mutual_features))
    mutual_features[features_already_selected] = 0
    mutual_features_idx = mutual_features.argsort()[::-1][:k]

    return mutual_features_idx

def correlation_select_k_features(X,y,k,features_already_selected = [],features_from_saved=False):
    if features_from_saved:
        return []
    k = min(k,X.shape[1]-len(features_already_selected))
    correlation_features = r_regression(X,y)
    correlation_features = np.abs(np.array(correlation_features))
    correlation_features[features_already_selected] = 0
    correlation_features_idx = correlation_features.argsort()[::-1][:k]

    return correlation_features_idx


   