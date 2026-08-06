"""
AI Personal Budget Planner
---------------------------
Intermediate-level project combining:
  1. A deterministic RULE-BASED CALCULATOR (50/30/20 budgeting rule,
     savings-goal projection, category overspend detection).
  2. An LLM CHAIN (Google Gemini API) that turns the calculator's
     numeric output into a natural-language financial coaching report.

Tech: Streamlit, Pandas, Google Gemini API (google-generativeai)
"""

import io
import json
from datetime import date, datetime

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Optional import: the app must still run (rule-based mode only) if the
# google-generativeai package or an API key is not available.
# --------------------------------------------------------------------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ==========================================================================
# CONSTANTS
# ==========================================================================
DEFAULT_CATEGORIES = [
    "Rent/Housing", "Groceries", "Transport", "Utilities",
    "Entertainment", "Healthcare", "Shopping", "Dining Out",
    "Subscriptions", "Other",
]

# 50/30/20 rule mapping: which categories count as Needs vs Wants
NEEDS = {"Rent/Housing", "Groceries", "Transport", "Utilities", "Healthcare"}
WANTS = {"Entertainment", "Shopping", "Dining Out", "Subscriptions", "Other"}

st.set_page_config(page_title="AI Personal Budget Planner", page_icon="💰", layout="wide")


# ==========================================================================
# SESSION STATE INITIALISATION
# ==========================================================================
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Description", "Amount"])

if "history" not in st.session_state:
    st.session_state.history = []  # list of past AI suggestion strings


# ==========================================================================
# RULE-BASED CALCULATOR
# ==========================================================================
def compute_budget_summary(income: float, expenses_df: pd.DataFrame,
                            savings_goal: float, goal_months: int) -> dict:
    """Pure rule-based financial calculator (no AI involved)."""
    total_expense = float(expenses_df["Amount"].sum()) if not expenses_df.empty else 0.0
    remaining = income - total_expense
    savings_rate = (remaining / income * 100) if income > 0 else 0.0

    # Category breakdown
    by_category = (
        expenses_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        if not expenses_df.empty else pd.Series(dtype=float)
    )

    # 50/30/20 rule targets
    needs_target = income * 0.50
    wants_target = income * 0.30
    savings_target = income * 0.20

    needs_actual = by_category[by_category.index.isin(NEEDS)].sum() if not by_category.empty else 0.0
    wants_actual = by_category[by_category.index.isin(WANTS)].sum() if not by_category.empty else 0.0

    # Overspend flags per category (simple rule: > 15% of income on one category = flag)
    overspend_flags = []
    for cat, amt in by_category.items():
        if income > 0 and amt > 0.15 * income:
            overspend_flags.append((cat, amt, amt / income * 100))

    # Savings goal projection
    monthly_savings_capacity = max(remaining, 0)
    months_to_goal = (savings_goal / monthly_savings_capacity) if monthly_savings_capacity > 0 else None
    on_track = (months_to_goal is not None) and (goal_months <= 0 or months_to_goal <= goal_months)

    return {
        "income": income,
        "total_expense": total_expense,
        "remaining": remaining,
        "savings_rate": savings_rate,
        "by_category": by_category,
        "needs_target": needs_target,
        "wants_target": wants_target,
        "savings_target": savings_target,
        "needs_actual": needs_actual,
        "wants_actual": wants_actual,
        "overspend_flags": overspend_flags,
        "savings_goal": savings_goal,
        "goal_months": goal_months,
        "monthly_savings_capacity": monthly_savings_capacity,
        "months_to_goal": months_to_goal,
        "on_track": on_track,
    }


def rule_based_tips(summary: dict) -> list:
    """Deterministic, non-AI fallback tips derived purely from the numbers."""
    tips = []
    if summary["income"] <= 0:
        return ["Enter your monthly income to generate a budget analysis."]

    if summary["remaining"] < 0:
        tips.append(
            f"⚠️ You are overspending by ₹{abs(summary['remaining']):,.2f} this month. "
            "Cut discretionary spending immediately."
        )
    elif summary["savings_rate"] < 20:
        tips.append(
            f"Your savings rate is {summary['savings_rate']:.1f}%, below the recommended 20%. "
            "Try trimming 'Wants' category spending."
        )
    else:
        tips.append(f"Great job! You're saving {summary['savings_rate']:.1f}% of your income.")

    if summary["needs_actual"] > summary["needs_target"]:
        tips.append(
            f"Your 'Needs' spending (₹{summary['needs_actual']:,.2f}) exceeds the 50% "
            f"guideline (₹{summary['needs_target']:,.2f})."
        )
    if summary["wants_actual"] > summary["wants_target"]:
        tips.append(
            f"Your 'Wants' spending (₹{summary['wants_actual']:,.2f}) exceeds the 30% "
            f"guideline (₹{summary['wants_target']:,.2f})."
        )

    for cat, amt, pct in summary["overspend_flags"]:
        tips.append(f"'{cat}' consumes {pct:.1f}% of your income (₹{amt:,.2f}) — consider a limit.")

    if summary["savings_goal"] > 0:
        if summary["months_to_goal"] is None:
            tips.append("At your current spending, you have no surplus to put toward your savings goal.")
        elif summary["on_track"]:
            tips.append(
                f"At this rate you'll hit your ₹{summary['savings_goal']:,.2f} goal in "
                f"~{summary['months_to_goal']:.1f} months — on track!"
            )
        else:
            tips.append(
                f"At this rate you'll need ~{summary['months_to_goal']:.1f} months to reach your goal, "
                f"which is later than your {summary['goal_months']}-month target. Increase savings or extend the timeline."
            )
    return tips


# ==========================================================================
# LLM CHAIN (Gemini) — turns the numeric summary into a coaching narrative
# ==========================================================================
def build_llm_prompt(summary: dict, expenses_df: pd.DataFrame) -> str:
    category_lines = "\n".join(
        f"- {cat}: ₹{amt:,.2f}" for cat, amt in summary["by_category"].items()
    ) or "No expenses recorded."

    prompt = f"""
You are a certified personal finance advisor AI. Analyze the user's monthly
budget data below and produce a concise, encouraging, and actionable report.

MONTHLY INCOME: ₹{summary['income']:,.2f}
TOTAL EXPENSES: ₹{summary['total_expense']:,.2f}
REMAINING BALANCE: ₹{summary['remaining']:,.2f}
CURRENT SAVINGS RATE: {summary['savings_rate']:.1f}%

EXPENSE BREAKDOWN BY CATEGORY:
{category_lines}

50/30/20 RULE CHECK:
- Needs: ₹{summary['needs_actual']:,.2f} (target ₹{summary['needs_target']:,.2f})
- Wants: ₹{summary['wants_actual']:,.2f} (target ₹{summary['wants_target']:,.2f})
- Savings: target ₹{summary['savings_target']:,.2f}

SAVINGS GOAL: ₹{summary['savings_goal']:,.2f} within {summary['goal_months']} months
PROJECTED MONTHS TO REACH GOAL AT CURRENT RATE: {summary['months_to_goal']}

Please respond with:
1. A short overall verdict (1-2 sentences).
2. Three specific, prioritized action items to improve the budget.
3. One tip tailored to the category with the highest spend.
4. A brief motivational closing line.

Keep the entire response under 200 words. Use simple, friendly language.
"""
    return prompt.strip()


def get_ai_suggestions(api_key: str, prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """Call the Gemini API (the 'LLM Chain' step) and return generated text."""
    if not GEMINI_AVAILABLE:
        return "google-generativeai package not installed. Run: pip install google-generativeai"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API call failed: {e}"


# ==========================================================================
# SIDEBAR — inputs
# ==========================================================================
st.sidebar.title("⚙️ Setup")

api_key = st.sidebar.text_input(
    "Gemini API Key", type="password",
    help="Get a free key at https://aistudio.google.com/app/apikey. "
         "Leave blank to use rule-based tips only."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Monthly Income & Goal")
income = st.sidebar.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=500.0)
savings_goal = st.sidebar.number_input("Savings Goal (₹)", min_value=0.0, value=100000.0, step=1000.0)
goal_months = st.sidebar.number_input("Target Timeline (months)", min_value=1, value=12, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Add an Expense")
with st.sidebar.form("expense_form", clear_on_submit=True):
    exp_date = st.date_input("Date", value=date.today())
    exp_category = st.selectbox("Category", DEFAULT_CATEGORIES)
    exp_desc = st.text_input("Description (optional)")
    exp_amount = st.number_input("Amount (₹)", min_value=0.0, step=50.0)
    submitted = st.form_submit_button("➕ Add Expense")
    if submitted and exp_amount > 0:
        new_row = pd.DataFrame([{
            "Date": exp_date, "Category": exp_category,
            "Description": exp_desc, "Amount": exp_amount,
        }])
        st.session_state.expenses = pd.concat(
            [st.session_state.expenses, new_row], ignore_index=True
        )
        st.sidebar.success("Expense added!")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Or upload expenses CSV", type=["csv"])
if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        expected = {"Date", "Category", "Description", "Amount"}
        if expected.issubset(df_upload.columns):
            st.session_state.expenses = pd.concat(
                [st.session_state.expenses, df_upload[list(expected)]], ignore_index=True
            )
            st.sidebar.success(f"Imported {len(df_upload)} rows.")
        else:
            st.sidebar.error(f"CSV must contain columns: {expected}")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

if st.sidebar.button("🗑️ Clear All Expenses"):
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Description", "Amount"])
    st.sidebar.success("Cleared.")


# ==========================================================================
# MAIN PAGE
# ==========================================================================
st.title("💰 AI Personal Budget Planner")
st.caption("LLM Chain + Rule-Based Calculator · Monthly expense tracking, savings goals, AI financial suggestions")

tab_overview, tab_expenses, tab_ai = st.tabs(["📊 Overview", "🧾 Expenses", "🤖 AI Suggestions"])

summary = compute_budget_summary(
    income, st.session_state.expenses, savings_goal, goal_months
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly Income", f"₹{summary['income']:,.0f}")
    col2.metric("Total Expenses", f"₹{summary['total_expense']:,.0f}")
    col3.metric("Remaining", f"₹{summary['remaining']:,.0f}",
                delta=f"{summary['savings_rate']:.1f}% savings rate")
    col4.metric("Savings Goal", f"₹{summary['savings_goal']:,.0f}")

    st.markdown("### Expense Breakdown by Category")
    if not summary["by_category"].empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.bar_chart(summary["by_category"])
        with c2:
            st.dataframe(
                summary["by_category"].reset_index().rename(
                    columns={"index": "Category", "Amount": "Total (₹)"}
                ),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("No expenses added yet. Use the sidebar to add your first expense.")

    st.markdown("### 50/30/20 Rule Check")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Needs (target 50%)", f"₹{summary['needs_actual']:,.0f}",
               delta=f"target ₹{summary['needs_target']:,.0f}")
    rc2.metric("Wants (target 30%)", f"₹{summary['wants_actual']:,.0f}",
               delta=f"target ₹{summary['wants_target']:,.0f}")
    rc3.metric("Savings (target 20%)", f"₹{summary['remaining']:,.0f}",
               delta=f"target ₹{summary['savings_target']:,.0f}")

    st.markdown("### 📌 Rule-Based Tips")
    for tip in rule_based_tips(summary):
        st.write("•", tip)

# ---------------------------------------------------------------- Expenses
with tab_expenses:
    st.markdown("### All Recorded Expenses")
    if st.session_state.expenses.empty:
        st.info("No expenses recorded yet.")
    else:
        edited_df = st.data_editor(
            st.session_state.expenses, num_rows="dynamic",
            use_container_width=True, key="expense_editor",
        )
        st.session_state.expenses = edited_df

        csv_buffer = io.StringIO()
        st.session_state.expenses.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download expenses as CSV", data=csv_buffer.getvalue(),
            file_name=f"expenses_{datetime.now():%Y%m%d}.csv", mime="text/csv",
        )

# ---------------------------------------------------------------- AI Tab
with tab_ai:
    st.markdown("### 🤖 AI-Generated Financial Suggestions")
    st.caption("This step sends your rule-based summary to Google Gemini (the LLM Chain) "
               "to generate a personalized natural-language report.")

    if st.button("✨ Generate AI Suggestions", type="primary"):
        if summary["income"] <= 0:
            st.warning("Please enter a monthly income first.")
        elif not api_key:
            st.warning("No Gemini API key provided — showing rule-based tips only.")
            for tip in rule_based_tips(summary):
                st.write("•", tip)
        else:
            with st.spinner("Contacting Gemini..."):
                prompt = build_llm_prompt(summary, st.session_state.expenses)
                result = get_ai_suggestions(api_key, prompt)
            st.session_state.history.append(result)
            st.success("AI report generated!")
            st.markdown(result)

    if st.session_state.history:
        with st.expander("📜 Past AI Reports"):
            for i, past in enumerate(reversed(st.session_state.history), 1):
                st.markdown(f"**Report {len(st.session_state.history) - i + 1}**")
                st.markdown(past)
                st.markdown("---")

st.markdown("---")
st.caption("Built with Streamlit, Pandas & Gemini API · Rule-based calculator ensures numbers are always accurate; "
           "the LLM only explains and advises.")
