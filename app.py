# loan_simulator_fixed.py
import math
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import json

# ---------------------------
# Helper functions
# ---------------------------
def calculate_emi(principal, annual_rate, months):
    r = annual_rate / (12 * 100)
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

def simulate_monthly_schedule(principal, annual_rate, emi, months, lumps=None):
    if lumps is None:
        lumps = {}
    r = annual_rate / (12 * 100)
    balance = principal
    rows = []
    for m in range(1, months + 1):
        interest = balance * r
        principal_comp = emi - interest
        if principal_comp > balance:
            principal_comp = balance
            emi = interest + principal_comp
        balance -= principal_comp
        if m in lumps:
            balance -= lumps[m]
            if balance < 0:
                balance = 0
        rows.append({
            "Month": m,
            "EMI": round(emi, 0),
            "Principal": round(principal_comp, 0),
            "Interest": round(interest, 0),
            "Balance": round(balance, 0)
        })
        if balance <= 0:
            break
    return pd.DataFrame(rows)

def simulate_with_income(principal, rate, months, base_emi, lumps, income, growth, emi_ratio):
    r = rate / (12 * 100)
    balance = principal
    rows = []
    emi = base_emi
    lumps_dict = {int(y) * 12: a for (y, a) in lumps if y and a}
    monthly_income = income
    for m in range(1, months * 2):
        if income and m % 12 == 1 and m > 1:
            monthly_income *= (1 + growth / 100)
            max_emi = monthly_income * emi_ratio / 100
            if emi > max_emi:
                emi = max_emi
        interest = balance * r
        principal_comp = emi - interest
        if principal_comp < 0:
            principal_comp = 0
        if principal_comp > balance:
            principal_comp = balance
            emi = interest + principal_comp
        balance -= principal_comp
        if m in lumps_dict:
            balance -= lumps_dict[m]
        if balance < 0:
            balance = 0
        rows.append({
            "Month": m,
            "EMI": round(emi, 0),
            "Principal": round(principal_comp, 0),
            "Interest": round(interest, 0),
            "Balance": round(balance, 0)
        })
        if balance <= 0:
            break
    return pd.DataFrame(rows)

def annualize(df):
    if df.empty:
        return df
    df2 = df.copy()
    df2["Year"] = ((df2["Month"] - 1) // 12) + 1
    ann = df2.groupby("Year").agg({"EMI":"sum","Principal":"sum","Interest":"sum"}).reset_index()
    ann["Total Payment"] = ann["EMI"]
    return ann.round(0)

# ---------------------------
# App setup
# ---------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

app.title = "Loan Simulator"

server = app.server

app.layout = dbc.Container([
    html.H2("🏦 Loan Repayment Strategy Simulator", style={"textAlign": "center", "marginTop": "10px"}),
    html.Hr(),

    # Store for simulation results (persist after run)
    dcc.Store(id="sim_store", data=None),

    dbc.Row([
        dbc.Col([
            html.H5("Borrower Inputs"),
            dbc.Label("Loan Amount (₹)"),
            dbc.Input(id="loan_amt", type="number", value=6000000, style={"marginBottom": "8px"}),

            dbc.Label("Monthly Income (₹)"),
            dbc.Input(id="income", type="number", value=100000, style={"marginBottom": "8px"}),

            dbc.Label("Expected Annual Income Growth (%)"),
            dbc.Input(id="income_growth", type="number", value=5.0, style={"marginBottom": "8px"}),

            dbc.Label("Desired EMI as % of Income (%)"),
            dbc.Input(id="emi_ratio", type="number", value=40, style={"marginBottom": "12px"}),

            html.H6("Interest Rates"),
            html.Div(id="rates_container", children=[
                dbc.Row([dbc.Col(dbc.Input(type="number", value=7.6, id={"type": "rate", "index": 0}), width=8)])
            ], style={"marginBottom": "8px"}),
            dbc.Button("Add Rate", id="add_rate", size="sm", color="secondary", style={"marginBottom": "12px"}),

            html.H6("Tenures (Years)"),
            html.Div(id="tenures_container", children=[
                dbc.Row([dbc.Col(dbc.Input(type="number", value=16, id={"type": "tenure", "index": 0}), width=8)])
            ], style={"marginBottom": "8px"}),
            dbc.Button("Add Tenure", id="add_tenure", size="sm", color="secondary", style={"marginBottom": "12px"}),

            html.H6("Lump-Sum Prepayments"),
            html.Div(id="lumps_container", children=[
                dbc.Row([dbc.Col(dbc.Input(type="number", placeholder="Year", value=5, id={"type": "lump_year", "index": 0}), width=4),
                         dbc.Col(dbc.Input(type="number", placeholder="Amount", value=500000, id={"type": "lump_amt", "index": 0}), width=8)])
            ], style={"marginBottom": "8px"}),
            dbc.Button("Add Lump-sum", id="add_lump", size="sm", color="secondary", style={"marginBottom": "12px"}),

            dbc.Button("Run Simulation", id="run_btn", color="success", style={"width": "100%"})
        ], width=4),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="🧾 Simulation Summary", tab_id="summary"),
                dbc.Tab(label="📑 Amortization Schedules", tab_id="amort"),
                dbc.Tab(label="📊 Strategy Plots", tab_id="plots"),
            ], id="tabs", active_tab="summary"),
            html.Div(id="tab_content", style={"marginTop": "15px"})
        ], width=8)
    ])
], fluid=True)

# ---------------------------
# Add inputs dynamically
# ---------------------------
@app.callback(
    Output("rates_container", "children"),
    Input("add_rate", "n_clicks"),
    State("rates_container", "children"),
    prevent_initial_call=True
)
def add_rate(n, children):
    idx = len(children)
    children.append(dbc.Row([dbc.Col(dbc.Input(type="number", placeholder="Rate (%)", id={"type": "rate", "index": idx}), width=8)]))
    return children

@app.callback(
    Output("tenures_container", "children"),
    Input("add_tenure", "n_clicks"),
    State("tenures_container", "children"),
    prevent_initial_call=True
)
def add_tenure(n, children):
    idx = len(children)
    children.append(dbc.Row([dbc.Col(dbc.Input(type="number", placeholder="Tenure (years)", id={"type": "tenure", "index": idx}), width=8)]))
    return children

@app.callback(
    Output("lumps_container", "children"),
    Input("add_lump", "n_clicks"),
    State("lumps_container", "children"),
    prevent_initial_call=True
)
def add_lump(n, children):
    idx = len(children)
    children.append(dbc.Row([
        dbc.Col(dbc.Input(type="number", placeholder="Year", id={"type":"lump_year","index":idx}), width=4),
        dbc.Col(dbc.Input(type="number", placeholder="Amount", id={"type":"lump_amt","index":idx}), width=8)
    ]))
    return children

# ---------------------------
# Callback 1: Run simulation and store results
# ---------------------------
@app.callback(
    Output("sim_store", "data"),
    Input("run_btn", "n_clicks"),
    State("loan_amt", "value"),
    State("income", "value"),
    State("income_growth", "value"),
    State("emi_ratio", "value"),
    State({"type": "rate", "index": ALL}, "value"),
    State({"type": "tenure", "index": ALL}, "value"),
    State({"type": "lump_year", "index": ALL}, "value"),
    State({"type": "lump_amt", "index": ALL}, "value"),
    prevent_initial_call=True
)
def run_simulation_store(n, loan_amt, income, growth, emi_ratio, rate_vals, tenure_vals, lump_years, lump_amts):
    rates = [float(r) for r in rate_vals if r is not None and str(r).strip() != ""]
    tenures = [int(t) for t in tenure_vals if t is not None and str(t).strip() != ""]
    lumps = []
    for y, a in zip(lump_years, lump_amts):
        if y is not None and a is not None and str(y).strip() != "" and str(a).strip() != "":
            lumps.append((int(y), float(a)))

    scenarios = []
    for rate in rates:
        for tenure in tenures:
            months = tenure * 12
            base_emi = calculate_emi(loan_amt, rate, months)
            df_base = simulate_monthly_schedule(loan_amt, rate, base_emi, months)
            df_inc = simulate_monthly_schedule(loan_amt, rate, base_emi * 1.1, months)
            df_prepay = simulate_with_income(loan_amt, rate, months, base_emi, lumps, income, growth, emi_ratio)

            scenarios.append({
                "rate": rate,
                "tenure": tenure,
                "baseline": df_base.to_dict("records"),
                "increase": df_inc.to_dict("records"),
                "save": df_prepay.to_dict("records")
            })

    summary_rows = []
    for s in scenarios:
        df_base = pd.DataFrame(s["baseline"])
        df_inc = pd.DataFrame(s["increase"])
        df_save = pd.DataFrame(s["save"])
        summary_rows.append({
            "rate": s["rate"],
            "tenure": s["tenure"],
            "baseline_interest": int(df_base["Interest"].sum()) if not df_base.empty else 0,
            "baseline_months": len(df_base),
            "increase_interest": int(df_inc["Interest"].sum()) if not df_inc.empty else 0,
            "increase_months": len(df_inc),
            "save_interest": int(df_save["Interest"].sum()) if not df_save.empty else 0,
            "save_months": len(df_save)
        })

    store = {
        "loan_amt": loan_amt,
        "income": income,
        "growth": growth,
        "emi_ratio": emi_ratio,
        "lumps": lumps,
        "scenarios": scenarios,
        "summary": summary_rows
    }
    return store  # dcc.Store will JSON-serialize

# ---------------------------
# Callback 2: Render tab content from stored results
# ---------------------------
@app.callback(
    Output("tab_content", "children"),
    Input("tabs", "active_tab"),
    Input("sim_store", "data"),
    prevent_initial_call=False
)
def render_tab(active_tab, store):
    if not store:
        return html.Div("No results yet. Enter inputs and click 'Run Simulation' to generate results.", style={"color":"gray"})

    scenarios = store.get("scenarios", [])
    summary_rows = store.get("summary", [])

    # Build summary table
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df_display = summary_df.rename(columns={
            "rate": "Rate (%)",
            "tenure": "Tenure (Years)",
            "baseline_interest": "Baseline Interest (₹)",
            "baseline_months": "Baseline Months",
            "increase_interest": "IncreaseEMI Interest (₹)",
            "increase_months": "IncreaseEMI Months",
            "save_interest": "SavePrepay Interest (₹)",
            "save_months": "SavePrepay Months"
        })
        summary_table = dbc.Table.from_dataframe(summary_df_display, striped=True, bordered=True, hover=True, responsive=True)
    else:
        summary_table = html.Div("No summary data")

    # Build amortization tabs (one tab per scenario)
    amort_tab_children = []
    for s in scenarios:
        label = f"{s['rate']}% - {s['tenure']}y"
        df_base = pd.DataFrame(s["baseline"])
        df_inc = pd.DataFrame(s["increase"])
        df_save = pd.DataFrame(s["save"])
        amort_tab_children.append(
            dbc.Tab(label=label, tab_id=f"amort_{s['rate']}_{s['tenure']}", children=[
                html.H6("Baseline (monthly)"),
                dbc.Table.from_dataframe(df_base.head(500), striped=True, bordered=True, hover=True, responsive=True),
                html.H6("Increase EMI (+10%) (monthly)"),
                dbc.Table.from_dataframe(df_inc.head(500), striped=True, bordered=True, hover=True, responsive=True),
                html.H6("Save & Prepay (monthly)"),
                dbc.Table.from_dataframe(df_save.head(500), striped=True, bordered=True, hover=True, responsive=True),
            ])
        )

    amort_tabs = dbc.Tabs(amort_tab_children) if amort_tab_children else html.Div("No amortization schedules")

    # Build plots area: cross-scenario total interest and per-scenario plots
    # Cross-scenario total interest
    bars = []
    for row in summary_rows:
        label = f"{row['rate']}% - {row['tenure']}y"
        bars.append(go.Bar(name=label, x=["Baseline", "IncreaseEMI", "SavePrepay"],
                           y=[row["baseline_interest"], row["increase_interest"], row["save_interest"]]))
    cross_fig = go.Figure(bars) if bars else None
    if cross_fig:
        cross_fig.update_layout(title="Total Interest by Strategy (per scenario)", barmode='group', template="plotly_white", height=420)
        cross_graph = dcc.Graph(figure=cross_fig)
    else:
        cross_graph = html.Div("No scenarios to plot")

    # Per-scenario detailed plots
    per_scenario_plots = []
    for s in scenarios:
        df_base = pd.DataFrame(s["baseline"])
        df_inc = pd.DataFrame(s["increase"])
        df_save = pd.DataFrame(s["save"])

        fig_bal = go.Figure()
        if not df_base.empty:
            fig_bal.add_trace(go.Scatter(x=df_base["Month"], y=df_base["Balance"], name="Baseline"))
        if not df_inc.empty:
            fig_bal.add_trace(go.Scatter(x=df_inc["Month"], y=df_inc["Balance"], name="Increase EMI"))
        if not df_save.empty:
            fig_bal.add_trace(go.Scatter(x=df_save["Month"], y=df_save["Balance"], name="Save & Prepay"))
        fig_bal.update_layout(title=f"Balance Trend ({s['rate']}%, {s['tenure']}y)", height=350, width=620, template="plotly_white")

        ann_base = annualize(df_base) if not df_base.empty else pd.DataFrame()
        ann_inc = annualize(df_inc) if not df_inc.empty else pd.DataFrame()
        ann_save = annualize(df_save) if not df_save.empty else pd.DataFrame()
        fig_ann = go.Figure()
        if not ann_base.empty:
            fig_ann.add_trace(go.Bar(x=ann_base["Year"], y=ann_base["Interest"], name="Baseline Interest"))
        if not ann_inc.empty:
            fig_ann.add_trace(go.Bar(x=ann_inc["Year"], y=ann_inc["Interest"], name="Increase EMI Interest"))
        if not ann_save.empty:
            fig_ann.add_trace(go.Bar(x=ann_save["Year"], y=ann_save["Interest"], name="Save Interest"))
        fig_ann.update_layout(title=f"Annual Interest ({s['rate']}%, {s['tenure']}y)", barmode="group", height=350, width=620, template="plotly_white")

        per_scenario_plots.append(html.Div([dcc.Graph(figure=fig_bal), dcc.Graph(figure=fig_ann)], style={"display":"inline-block", "marginRight":"14px"}))

    # Return content for the requested tab
    if active_tab == "summary":
        return html.Div([html.H6("Scenario Summary"), summary_table])
    elif active_tab == "amort":
        return html.Div([
            html.H6("Amortization Schedules (select scenario tab)"),
            amort_tabs
        ])
    else:  # plots
        return html.Div([html.H6("Cross-scenario Total Interest Comparison"), cross_graph, html.Hr()] + per_scenario_plots, style={"whiteSpace":"nowrap","overflowX":"auto"})

# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    app.run(port=8050)