import shap
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

if 'model1' not in st.session_state:
    model1 = joblib.load('clf.pkl')
    model2 = joblib.load('clf2.pkl')
    st.session_state["model1"] = model1
    st.session_state["model2"] = model2
else:
    model1 = st.session_state["model1"]
    model2 = st.session_state["model2"]
scaler = joblib.load("Scaler.pkl")
continuous_vars = ['age', 'aniongap', 'resp_rate', 'apsiii']

st.set_page_config(layout='wide')

st.write("<h1 style='text-align: center'>Analysis of device-associated infections and survival</h1>",
         unsafe_allow_html=True)

dic1 = {
    '男': 1,
    '女': 0
}
dic2 = {
    'Yes': 1,
    'No': 0
}
with st.sidebar:
    st.markdown("# Input variable")
    st.markdown('-----')
    mv = st.selectbox("mv", ["Yes", "No"])
    uc = st.selectbox("uc", ["Yes", "No"])
    cvc = st.selectbox("cvc", ["Yes", "No"])
    age = st.text_input("age")
    gender = st.selectbox("gender", ["男", "女"])
    aniongap = st.text_input("aniongap")
    resp_rate = st.text_input("resp_rate")
    liver_disease = st.selectbox("liver_disease", ["Yes", "No"])
    myocardial_infarct = st.selectbox("myocardial_infarct", ["Yes", "No"])
    malignant_cancer = st.selectbox("malignant_cancer", ["Yes", "No"])
    tracheostomy = st.selectbox("tracheostomy", ["Yes", "No"])
    apsiii = st.text_input("apsiii")
    race_black = st.selectbox("race_black", ["Yes", "No"])
    case_micu_sicu = st.selectbox("case_micu/sicu", ["Yes", "No"])
if st.sidebar.button("Predict"):
    with st.spinner("Forecast, please wait..."):
        test_df = pd.DataFrame([dic2[mv], dic2[uc], dic2[cvc], dic1[gender],
                                dic2[liver_disease], dic2[myocardial_infarct], dic2[malignant_cancer],
                                dic2[case_micu_sicu]],
                               index=['mv', 'uc', 'cvc', 'gender', 'liver_disease', 'myocardial_infarct',
                                      'malignant_cancer', 'case_micu/sicu']).T
        explainer = shap.Explainer(model1)  # 创建解释器
        shap_ = explainer.shap_values(test_df)

        shap.waterfall_plot(
            shap.Explanation(values=shap_[0, :], base_values=explainer.expected_value, data=test_df.iloc[0, :]),
            show=False)
        plt.tight_layout()
        plt.savefig('shap1.png', dpi=300)
        col1, col2, col3 = st.columns([2, 5, 3])
        shap.initjs()

        shap.force_plot(explainer.expected_value, shap_[0, :], test_df.iloc[0, :], show=False, matplotlib=True,
                        figsize=(20, 5))
        plt.xticks(fontproperties='Times New Roman', size=16)
        plt.yticks(fontproperties='Times New Roman', size=16)
        plt.tight_layout()
        plt.savefig('shap2.png', dpi=300)
        with col2:
            st.image('shap1.png')
        st.image('shap2.png')
        st.success("Probability: {:.3f}%".format(model1.predict_proba(test_df)[:, 1][0] * 100))
        st.markdown('----')
        test_df2 = pd.DataFrame(
            [dic2[mv], float(age), float(aniongap), float(resp_rate), dic2[tracheostomy], float(apsiii),
             dic2[race_black],
             dic2[case_micu_sicu]],
            index=['mv', 'age', 'aniongap', 'resp_rate', 'tracheostomy', 'apsiii',
                   'race_black', 'case_micu/sicu']).T
        test_df2[continuous_vars] = scaler.transform(test_df2[continuous_vars])
        surv_funcs = model2.predict_survival_function(test_df2)
        col4, col5, col6 = st.columns([2, 5, 3])
        fig, ax = plt.subplots()
        for fn in surv_funcs:
            ax.step(fn.x, fn(fn.x), where="post", color="#8dd3c7", lw=2)
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $days$")
        plt.grid()
        with col5:
            st.pyplot(fig)
