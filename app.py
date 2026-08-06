"""
AI Personal Budget Planner
---------------------------
Intermediate-level project combining:
  1. A deterministic RULE-BASED CALCULATOR (50/30/20 budgeting rule,
     savings-goal projection, category overspend detection).
  2. An LLM CHAIN (Google Gemini API) that turns the calculator's
     numeric output into a natural-language financial coaching report.
  3. A PDF EXPORT that bundles income/goal, expenses, the 50/30/20 check,
     rule-based tips, and the latest AI report into one downloadable file.

Tech: Streamlit, Pandas, Google Gemini API (google-generativeai), ReportLab
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

# --------------------------------------------------------------------------
# ReportLab for PDF export
# --------------------------------------------------------------------------
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER


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
# PDF EXPORT — bundles income/goal, expenses, 50/30/20 check, tips & AI report
# ==========================================================================
def _rupee(value) -> str:
    """Format currency safely for ReportLab (avoids unicode glyph issues)."""
    try:
        return f"Rs. {value:,.2f}"
    except (TypeError, ValueError):
        return "Rs. 0.00"


def generate_pdf_report(summary: dict, expenses_df: pd.DataFrame,
                         ai_text: str | None = None) -> bytes:
    """Builds a single PDF containing the full budget report and returns it as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#1a5632"),
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCustom", parent=styles["Normal"], alignment=TA_CENTER,
        textColor=colors.grey, fontSize=10, spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], textColor=colors.HexColor("#1a5632"),
        spaceBefore=14, spaceAfter=8,
    )
    body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=10, leading=14)
    bullet_style = ParagraphStyle(
        "BulletCustom", parent=styles["Normal"], fontSize=10, leading=14,
        leftIndent=14, bulletIndent=0, spaceAfter=4,
    )

    story = []

    # ---- Header ----
    story.append(Paragraph("💰 AI Personal Budget Planner — Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now():%d %B %Y, %I:%M %p}", subtitle_style
    ))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#1a5632"), thickness=1))
    story.append(Spacer(1, 12))

    # ---- Income & Goal Summary ----
    story.append(Paragraph("Income & Savings Goal", h2_style))
    summary_table_data = [
        ["Monthly Income", _rupee(summary["income"])],
        ["Total Expenses", _rupee(summary["total_expense"])],
        ["Remaining Balance", _rupee(summary["remaining"])],
        ["Savings Rate", f"{summary['savings_rate']:.1f}%"],
        ["Savings Goal", _rupee(summary["savings_goal"])],
        ["Target Timeline", f"{summary['goal_months']} months"],
        [
            "Projected Months to Goal",
            f"{summary['months_to_goal']:.1f} months" if summary["months_to_goal"] is not None else "N/A (no surplus)",
        ],
        ["On Track?", "Yes" if summary["on_track"] else "No"],
    ]
    t = Table(summary_table_data, colWidths=[2.6 * inch, 3.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef5f0")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a5632")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # ---- 50/30/20 Rule Check ----
    story.append(Paragraph("50/30/20 Rule Check", h2_style))
    rule_table_data = [
        ["Category", "Actual", "Target"],
        ["Needs (50%)", _rupee(summary["needs_actual"]), _rupee(summary["needs_target"])],
        ["Wants (30%)", _rupee(summary["wants_actual"]), _rupee(summary["wants_target"])],
        ["Savings (20%)", _rupee(summary["remaining"]), _rupee(summary["savings_target"])],
    ]
    t2 = Table(rule_table_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5632")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)

    # ---- Expense Breakdown by Category ----
    story.append(Paragraph("Expense Breakdown by Category", h2_style))
    if not summary["by_category"].empty:
        cat_data = [["Category", "Amount", "% of Income"]]
        for cat, amt in summary["by_category"].items():
            pct = (amt / summary["income"] * 100) if summary["income"] > 0 else 0
            cat_data.append([cat, _rupee(amt), f"{pct:.1f}%"])
        t3 = Table(cat_data, colWidths=[2.5 * inch, 2 * inch, 1.5 * inch])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5632")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t3)
    else:
        story.append(Paragraph("No expenses recorded.", body_style))

    # ---- Detailed Expense Log ----
    story.append(Paragraph("Detailed Expense Log", h2_style))
    if not expenses_df.empty:
        log_data = [["Date", "Category", "Description", "Amount"]]
        for _, row in expenses_df.iterrows():
            log_data.append([
                str(row.get("Date", "")),
                str(row.get("Category", "")),
                str(row.get("Description", "") or "-"),
                _rupee(row.get("Amount", 0)),
            ])
        t4 = Table(log_data, colWidths=[1 * inch, 1.6 * inch, 2.4 * inch, 1 * inch], repeatRows=1)
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5632")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t4)
    else:
        story.append(Paragraph("No expenses recorded.", body_style))

    # ---- Rule-Based Tips ----
    story.append(Paragraph("Rule-Based Tips", h2_style))
    for tip in rule_based_tips(summary):
        clean_tip = tip.replace("⚠️", "[!]")
        story.append(Paragraph(f"&bull; {clean_tip}", bullet_style))

    # ---- AI Suggestions ----
    story.append(Paragraph("AI Financial Coaching Report", h2_style))
    if ai_text:
        clean_ai_text = ai_text.replace("⚠️", "[!]")
        for para in clean_ai_text.split("\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, body_style))
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph(
            "No AI report was generated for this session. Generate one in the "
            "'AI Suggestions' tab before exporting to include it here.",
            body_style,
        ))

    # ---- Footer note ----
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=0.5))
    story.append(Paragraph(
        "Built with Streamlit, Pandas & Gemini API. Rule-based calculator ensures numbers "
        "are always accurate; the LLM only explains and advises.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=8),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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

tab_overview, tab_expenses, tab_ai, tab_export = st.tabs(
    ["📊 Overview", "🧾 Expenses", "🤖 AI Suggestions", "📄 Export PDF"]
)

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

# ---------------------------------------------------------------- Export Tab
with tab_export:
    st.markdown("### 📄 Download Full Report as PDF")
    st.caption(
        "Bundles your monthly income, savings goal, full expense log, category "
        "breakdown, 50/30/20 check, rule-based tips, and the latest AI coaching "
        "report (if generated) into a single PDF file."
    )

    if summary["income"] <= 0:
        st.warning("Enter a monthly income in the sidebar before generating a PDF report.")
    else:
        latest_ai_report = st.session_state.history[-1] if st.session_state.history else None

        if latest_ai_report:
            st.info("The most recently generated AI report will be included in the PDF.")
        else:
            st.info(
                "No AI report has been generated yet — the PDF will still include all "
                "rule-based numbers and tips. Visit the 'AI Suggestions' tab first if "
                "you'd like the AI narrative included."
            )

        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Building PDF..."):
                pdf_bytes = generate_pdf_report(
                    summary, st.session_state.expenses, ai_text=latest_ai_report
                )
            st.success("PDF ready!")
            st.download_button(
                "⬇️ Download Budget Report (PDF)",
                data=pdf_bytes,
                file_name=f"budget_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
            )

st.markdown("---")
st.caption("Built with Streamlit, Pandas & Gemini API · Rule-based calculator ensures numbers are always accurate; "
           "the LLM only explains and advises.")
