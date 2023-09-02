# 导入需要的库
import streamlit as st
import pandas as pd
import joblib

st.header("Helicobacter pylori Multidrug Resistance Prediction Based on a Machine Learning Model")
st.sidebar.header('Variables')

# 选择框
a = st.sidebar.selectbox("rlpA Thr216Lys", ("presence", "absence"))
b = st.sidebar.selectbox("group_1364", ("presence", "absence"))
c = st.sidebar.selectbox("glmU Glu162Thr", ("presence", "absence"))
d = st.sidebar.selectbox("HP_0731 Asn511Asp", ("presence", "absence"))
e = st.sidebar.selectbox("smc", ("presence", "absence"))
f = st.sidebar.selectbox("gspA", ("presence", "absence"))
g = st.sidebar.selectbox("group_333", ("presence", "absence"))
h = st.sidebar.selectbox("polA Val112Thr", ("presence", "absence"))
i = st.sidebar.selectbox("omp13 SerLeu10PhePhe", ("presence", "absence"))
j = st.sidebar.selectbox("HP_0922 Ser2141Ala", ("presence", "absence"))


# 如果按下按钮
if st.button("Predict"):  # 显示按钮
    # 加载训练好的模型
    model = joblib.load("XGBoost.pkl")
    # 将输入存储DataFrame
    X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j]],
                     columns=['rlpA Thr216Lys', 'group_1364', 'glmU Glu162Thr', 'HP_0731 Asn511Asp', 'smc', 'gspA', 'group_333', 'polA Val112Thr', 'omp13 SerLeu10PhePhe', 'HP_0922 Ser2141Ala'])

    X = X.replace(["presence", "absence"], [1, 0])


    # 进行预测
    prediction = model.predict(X)[0]
    Predict_proba = model.predict_proba(X)[:, 1][0]
    # 输出预测结果
    if prediction == 0:
        st.subheader(f"The predicted result of Hp MDR:  Sensitive")
    else:
        st.subheader(f"The predicted result of Hp MDR:  Resistant")
    # 输出概率
    st.subheader(f"The probability of Hp MDR:  {'%.2f' % float(Predict_proba * 100) + '%'}")
