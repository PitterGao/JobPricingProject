# app.py
import numpy as np
import streamlit as st

from src.predict import load_predictor, predict_one

st.set_page_config(page_title="企业岗位定价模型 Demo", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def _load(tag: str):
    """加载对应版本的推理器（会缓存）"""
    return load_predictor(price_tag=tag)


def _health_sensitivity(predictor, base_payload, steps=21):
    """
    对 job_health_score 做扫描，观察推荐价格随健康度变化趋势
    """
    hs = np.linspace(0.0, 1.0, steps)
    prices = []
    for h in hs:
        p = dict(base_payload)
        p["job_health_score"] = float(h)
        out = predict_one(predictor, p)
        prices.append(out["pred_price"])
    return hs, np.array(prices, dtype=float)


def _variant_label(tag: str) -> str:
    mapping = {
        "none": "None",
        "lstm": "LSTM",
        "transformer": "Transformer",
    }
    return mapping.get(tag, tag)


# -----------------------------
# UI
# -----------------------------
st.title("企业岗位定价模型演示系统")
st.caption("本页面用于展示：企业分层（Tier）、岗位曝光预测（Impressions）与岗位定价推荐（Price）的端到端 baseline 推理结果。")

with st.sidebar:
    st.header("模型配置")
    price_tag = st.selectbox(
        "定价模型版本",
        ["none", "lstm", "transformer"],
        index=0,
        format_func=_variant_label,
        help="Baseline：不使用健康度特征；LSTM/Transformer：使用对应健康度特征训练的定价模型。",
    )

    st.divider()
    st.header("企业信息输入")
    company_size = st.number_input("企业规模（Company Size）", min_value=10, max_value=20000, value=300, step=10)
    brand_level = st.slider("品牌等级（Brand Level，1~5）", min_value=1, max_value=5, value=3)
    funding_stage = st.selectbox("融资阶段（Funding Stage）", ["seed", "A", "B", "C", "D", "public"], index=2)
    intl_flag = st.selectbox("是否国际化（International Flag）", [0, 1], index=0, help="0=否，1=是")

    st.divider()
    st.header("岗位信息输入")
    job_function = st.selectbox("岗位方向（Job Function）",
                                ["backend", "frontend", "data", "algo", "product", "qa", "ops"], index=0)
    job_level = st.selectbox("岗位级别（Job Level）", ["junior", "mid", "senior", "lead"], index=1)
    location = st.selectbox("工作地点（Location）", ["sydney", "melbourne", "remote", "shanghai", "beijing"], index=0)
    salary_min = st.number_input("薪资下限（Salary Min）", min_value=30000, max_value=400000, value=90000, step=1000)
    salary_max = st.number_input("薪资上限（Salary Max）", min_value=30000, max_value=450000, value=130000, step=1000)
    target_top10 = st.selectbox("是否面向 Top10（Target Top10）", [0, 1], index=0, help="0=否，1=是")

    st.divider()
    st.header("软性指标（0~100）")
    training_budget = st.slider("培训投入（Training Budget）", 0, 100, 60)
    promotion_rate = st.slider("晋升比例（Promotion Rate）", 0, 100, 55)
    work_life_balance = st.slider("工作生活平衡（Work-life Balance）", 0, 100, 58)
    hr_response_speed = st.slider("HR 响应速度（HR Response Speed）", 0, 100, 62)
    benefits_score = st.slider("福利水平（Benefits Score）", 0, 100, 57)

    st.divider()
    st.header("岗位健康度（Job Health Score）")
    st.caption("此处支持手动输入以观察定价敏感性：")
    job_health_score = st.slider("健康度分数（0~1）", 0.0, 1.0, 0.50, 0.01)

    st.divider()
    run_btn = st.button("开始预测 / 生成推荐", type="primary")

# Load predictor according to selected pricing model variant
predictor = _load(price_tag)

payload = {
    "company_size": company_size,
    "brand_level": brand_level,
    "funding_stage": funding_stage,
    "intl_flag": intl_flag,
    "job_function": job_function,
    "job_level": job_level,
    "location": location,
    "salary_min": salary_min,
    "salary_max": salary_max,
    "target_top10": target_top10,
    "training_budget": training_budget,
    "promotion_rate": promotion_rate,
    "work_life_balance": work_life_balance,
    "hr_response_speed": hr_response_speed,
    "benefits_score": benefits_score,
    "job_health_score": float(job_health_score),
}

# -----------------------------
# Main area
# -----------------------------
if run_btn:
    out = predict_one(predictor, payload)

    st.subheader("预测结果概览")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("定价模型版本", _variant_label(price_tag))
    c2.metric("企业档位（Enterprise Tier）", out["enterprise_tier"])
    c3.metric("人才关怀指数（TCI）", f"{out['talent_care_index']:.3f}")
    c4.metric("预测曝光（Impressions）", f"{out['pred_impressions']:.1f}")

    st.subheader("推荐定价（Baseline Output）")
    st.metric("预测价格（Predicted Price）", f"${out['pred_price']:.0f}")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("人才关怀指数拆解（加权贡献）")
        st.bar_chart(out["talent_care_breakdown"])

        st.subheader("输入信息回显")
        st.json({
            "模型版本": _variant_label(price_tag),
            "企业规模": company_size,
            "品牌等级": brand_level,
            "融资阶段": funding_stage,
            "是否国际化": intl_flag,
            "岗位方向": job_function,
            "岗位级别": job_level,
            "工作地点": location,
            "薪资下限": salary_min,
            "薪资上限": salary_max,
            "是否面向Top10": target_top10,
            "健康度分数": float(job_health_score),
        })

    with right:
        st.subheader("健康度敏感性分析：价格 vs 健康度")
        hs, prices = _health_sensitivity(predictor, payload, steps=21)
        st.line_chart({"健康度分数": hs, "预测价格": prices})

        st.caption(
            "注：用于评估定价模型对岗位健康度特征的敏感性。"
            "Baseline（none）通常响应较弱；LSTM/Transformer 版本理论上更敏感，具体以模型训练效果与特征质量为准。"
        )

    st.divider()
    st.caption(
        "注：本演示系统为 baseline 实现（模拟数据 + 曝光预测 + 定价回归），"
        "并支持健康度特征（LSTM/Transformer 可选）。"
    )

else:
    st.info("请在左侧填写输入信息，选择定价模型版本，然后点击「开始预测 / 生成推荐」。")
