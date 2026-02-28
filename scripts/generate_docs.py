"""
Repository Documentation Generator
====================================
Produces a comprehensive PDF audit of every file in the adaptive-portfolio-rl
codebase, how each file connects to the others, and how the full system works
from data ingest to result output.

Author: Prince Alexander John
Date: February 2026
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime
import pathlib


# ---------------------------------------------------------
#  Colour palette
# ---------------------------------------------------------
C_NAVY    = (15,  40,  80)    # section headers
C_TEAL    = (0,  120, 130)    # sub-headers
C_AMBER   = (180, 100,  0)    # highlight / callout
C_LTGRAY  = (245, 245, 245)   # table row fill
C_MIDGRAY = (200, 200, 200)   # borders
C_DKGRAY  = (80,  80,  80)    # body text
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)


class DocPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=20, top=20, right=20)
        self._toc = []          # (title, page)
        self._section_num = 0
        self._subsection_num = 0

    # -- header / footer ----------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_DKGRAY)
        self.cell(0, 6, "Regime-Aware Experience Replay  |  Repository Audit", align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_MIDGRAY)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_DKGRAY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # -- cover --------------------------------------------
    def cover(self):
        self.add_page()
        # Navy banner
        self.set_fill_color(*C_NAVY)
        self.rect(0, 0, 210, 80, "F")
        self.set_y(20)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*C_WHITE)
        self.multi_cell(0, 10, "Regime-Aware Experience Replay\nfor Deep Reinforcement Learning\nin Adaptive Portfolio Management",
                        align="C")
        self.ln(4)
        self.set_font("Helvetica", "I", 12)
        self.cell(0, 8, "Complete Repository Audit & Technical Reference", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(90)
        self.set_text_color(*C_DKGRAY)
        meta = [
            ("Author",      "Prince Alexander John"),
            ("Institution", "University of Birmingham"),
            ("Module",      "Final Year Project (MSc Computer Science)"),
            ("Duration",    "January 2026 - March 2026  |  Deadline: 27 March 2026"),
            ("Generated",   datetime.now().strftime("%d %B %Y  %H:%M")),
        ]
        self.set_font("Helvetica", "", 11)
        for label, val in meta:
            self.set_font("Helvetica", "B", 11)
            self.cell(45, 8, label + ":", new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.set_font("Helvetica", "", 11)
            self.cell(0, 8, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Teal divider
        self.set_draw_color(*C_TEAL)
        self.set_line_width(1.0)
        self.line(20, self.get_y() + 4, 190, self.get_y() + 4)
        self.ln(12)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*C_NAVY)
        self.cell(0, 8, "Document Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_DKGRAY)
        sections = [
            "1.  Project Overview & Novel Contribution",
            "2.  Repository File Map",
            "3.  Module: Data Pipeline  (src/data/)",
            "4.  Module: Trading Environment  (src/environment/)",
            "5.  Module: Regime Detection  (src/regime/)",
            "6.  Module: Replay Buffers  (src/replay/)",
            "7.  Module: Agents  (src/agents/)",
            "8.  Module: Training Infrastructure  (src/training/)",
            "9.  Module: Analysis  (src/analysis/)",
            "10. Module: Utilities  (src/utils/)",
            "11. Entry-Point Scripts  (scripts/)",
            "12. Configuration Files  (configs/)",
            "13. End-to-End Data Flow Walkthrough",
            "14. File Dependency Graph",
            "15. Experimental Design",
            "16. Financial Metrics Reference",
        ]
        for s in sections:
            self.cell(0, 6, "    " + s, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # -- helpers ------------------------------------------
    def section(self, title: str):
        self._section_num += 1
        self._subsection_num = 0
        self.add_page()
        self._toc.append((f"{self._section_num}. {title}", self.page_no()))
        self.set_fill_color(*C_NAVY)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, f"  {self._section_num}.  {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)
        self.set_text_color(*C_DKGRAY)

    def subsection(self, title: str):
        self._subsection_num += 1
        label = f"{self._section_num}.{self._subsection_num}  {title}"
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*C_TEAL)
        self.cell(0, 8, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_TEAL)
        self.set_line_width(0.5)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(2)
        self.set_text_color(*C_DKGRAY)

    def h3(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_NAVY)
        self.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*C_DKGRAY)

    def body(self, text: str):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_DKGRAY)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code(self, text: str):
        """Monospace code block with grey background."""
        self.set_fill_color(*C_LTGRAY)
        self.set_draw_color(*C_MIDGRAY)
        self.set_font("Courier", "", 8)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.5, text, border=1, fill=True)
        self.set_text_color(*C_DKGRAY)
        self.set_font("Helvetica", "", 9.5)
        self.ln(1)

    def callout(self, label: str, text: str):
        """Amber highlight box for key concepts."""
        self.set_fill_color(255, 248, 220)
        self.set_draw_color(*C_AMBER)
        self.set_line_width(0.8)
        x, y = self.get_x(), self.get_y()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_AMBER)
        self.cell(0, 6, f"  {label}", fill=True, border="LTR",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 40, 0)
        self.multi_cell(0, 5, f"  {text}", border="LBR", fill=True)
        self.set_text_color(*C_DKGRAY)
        self.set_line_width(0.2)
        self.ln(2)

    def table(self, headers: list, rows: list, col_widths: list = None):
        """Draw a formatted table."""
        total_w = 170
        if col_widths is None:
            w = total_w // len(headers)
            col_widths = [w] * len(headers)

        # Header row
        self.set_fill_color(*C_NAVY)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, f" {h}", border=1, fill=True,
                      new_x=XPos.RIGHT, new_y=YPos.LAST)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        fill = False
        for row in rows:
            self.set_fill_color(*(C_LTGRAY if fill else C_WHITE))
            self.set_text_color(*C_DKGRAY)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, f" {cell}", border=1, fill=True,
                          new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.ln()
            fill = not fill
        self.ln(2)

    def bullet(self, items: list, indent: int = 4):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_DKGRAY)
        for item in items:
            self.set_x(20 + indent)
            self.cell(4, 5.5, chr(149), new_x=XPos.RIGHT, new_y=YPos.LAST)
            self.multi_cell(0, 5.5, item)


# ---------------------------------------------------------
#  Build document
# ---------------------------------------------------------

def build(pdf: DocPDF):

    pdf.cover()

    # ====================================================
    # 1. PROJECT OVERVIEW
    # ====================================================
    pdf.section("Project Overview & Novel Contribution")

    pdf.subsection("What This Project Does")
    pdf.body(
        "This project implements and evaluates Regime-Aware Experience Replay -- a novel "
        "modification to the learning mechanism of off-policy deep reinforcement learning "
        "agents applied to multi-asset portfolio management.\n\n"
        "A portfolio management agent must allocate capital across 5 assets (SPY, QQQ, "
        "TLT, GLD, BIL) to maximise risk-adjusted returns. The key research insight is "
        "that financial markets cycle through distinct 'regimes' -- Bull (rising, calm), "
        "Bear (falling, volatile), and Sideways (no trend) -- and that standard RL training "
        "is biased toward whichever regime dominates the training period."
    )

    pdf.callout(
        "THE NOVEL CONTRIBUTION",
        "Prior work (Nixon Raj 2025) adds regime information to what the agent SEES "
        "(the observation). This project instead modifies HOW the agent LEARNS by "
        "changing the replay buffer's sampling mechanism. Even when an agent can see "
        "regime labels, standard uniform replay still biases learning toward frequent "
        "regimes. Our RegimeAwareReplayBuffer ensures the agent learns equally from ALL "
        "market conditions regardless of their historical frequency."
    )

    pdf.subsection("Prior Work vs This Project")
    pdf.table(
        ["Aspect", "Prior Work (Observation-Based)", "This Project (Replay-Based)"],
        [
            ["Where regime used", "In the state/observation space", "In the replay buffer sampling"],
            ["What it affects", "What the agent sees", "How the agent learns"],
            ["Mechanism", "Regime one-hot appended to obs", "Intelligent batch sampling"],
            ["Core modification", "Network input layer", "Learning algorithm"],
            ["Limitation addressed", "Agent sees regime", "Agent trains equally on all regimes"],
        ],
        col_widths=[50, 60, 60]
    )

    pdf.subsection("Four Replay Sampling Strategies")
    pdf.body("The novel contribution is implemented as four distinct strategies inside "
             "src/replay/regime_aware_buffer.py:")
    pdf.table(
        ["Strategy", "Sampling Rule", "Research Hypothesis"],
        [
            ["standard", "Uniform random (baseline)", "H0: no effect (control)"],
            ["balanced", "P(regime) = 1/3 for each", "H1: equal learning across regimes"],
            ["current_focused", "P(current)=0.6, others=0.2 each", "H4: focus on present regime"],
            ["inverse_frequency", "P(regime) prop. 1/count(regime)", "H3: oversample rare regimes"],
            ["adaptive", "P(regime) prop. mean(TD_error)", "Focus on hardest regime"],
        ],
        col_widths=[42, 70, 58]
    )

    pdf.subsection("Research Hypotheses")
    pdf.table(
        ["ID", "Hypothesis", "Test"],
        [
            ["H1", "Regime-aware replay improves Sharpe vs standard", "E1 (100 runs)"],
            ["H2", "Balanced sampling reduces variance across regimes", "E1 + E2"],
            ["H3", "Inverse-frequency improves bear market performance", "E2 regime breakdown"],
            ["H4", "Current-focused improves short-term adaptability", "E2 time-series"],
            ["H5", "Benefits are consistent across all 4 algorithms", "E1 cross-algo"],
            ["H6", "Robust to up to 20% noise in regime labels", "E3 robustness"],
        ],
        col_widths=[12, 100, 58]
    )

    pdf.subsection("Asset Universe & Date Splits")
    pdf.table(
        ["Asset", "Ticker", "Role", "Rationale"],
        [
            ["US Large Cap", "SPY", "S&P 500 benchmark", "Core equity holding"],
            ["US Technology", "QQQ", "Nasdaq 100 ETF", "Growth / risk-on"],
            ["US Long Bonds", "TLT", "20+ yr Treasury ETF", "Flight-to-safety / rates"],
            ["Gold", "GLD", "Gold ETF", "Inflation hedge"],
            ["Cash", "BIL", "T-Bill ETF", "Risk-free alternative"],
        ],
        col_widths=[35, 20, 55, 60]
    )
    pdf.table(
        ["Split", "Date Range", "Trading Days", "Purpose"],
        [
            ["Training", "2010-01-01 to 2019-12-31", "2,515", "Agent learns policy"],
            ["Validation", "2020-01-01 to 2021-12-31", "504", "Generalisation check"],
            ["Test (PRIMARY)", "2022-01-01 to 2023-12-31", "501", "Reported results only"],
        ],
        col_widths=[40, 55, 30, 45]
    )

    # ====================================================
    # 2. REPOSITORY FILE MAP
    # ====================================================
    pdf.section("Repository File Map")

    pdf.subsection("Directory Structure")
    pdf.code(
        "adaptive-portfolio-rl/\n"
        "+-- src/                          # All source code\n"
        "|   +-- agents/                   # 4 RL algorithms + base interface\n"
        "|   |   +-- __init__.py           # AGENTS dict, exports\n"
        "|   |   +-- base_agent.py         # BaseAgent, TrainingConfig, ReplayBufferWrapper\n"
        "|   |   +-- ppo_agent.py          # PPOAgent + RegimeObservationWrapper\n"
        "|   |   +-- sac_agent.py          # SACAgent + RegimeAwareReplayBufferWrapper\n"
        "|   |   +-- td3_agent.py          # TD3Agent\n"
        "|   |   +-- rainbow_agent.py      # RainbowAgent (custom training loop)\n"
        "|   +-- data/\n"
        "|   |   +-- downloader.py         # Yahoo Finance download + CSV cache\n"
        "|   |   +-- features.py           # 19 technical indicators per asset\n"
        "|   +-- environment/\n"
        "|   |   +-- trading_env.py        # Gymnasium portfolio environment\n"
        "|   +-- regime/\n"
        "|   |   +-- __init__.py           # UnifiedRegimeDetector, MarketRegime\n"
        "|   |   +-- rule_based_detector.py  # SMA crossover + volatility rules\n"
        "|   |   +-- hmm_detector.py       # Hidden Markov Model detector\n"
        "|   +-- replay/\n"
        "|   |   +-- __init__.py           # Re-exports\n"
        "|   |   +-- standard_buffer.py    # Baseline uniform replay buffer\n"
        "|   |   +-- regime_aware_buffer.py  # THE NOVEL CONTRIBUTION\n"
        "|   +-- training/\n"
        "|   |   +-- __init__.py\n"
        "|   |   +-- run_config.py         # RunConfig + 6 experiment generators\n"
        "|   |   +-- agent_factory.py      # create_agent() factory function\n"
        "|   |   +-- experiment_runner.py  # Manifest, ExperimentRunner, execute_single_run\n"
        "|   |   +-- baselines.py          # Buy-and-hold, equal-weight, 60/40, momentum\n"
        "|   +-- analysis/\n"
        "|   |   +-- results_loader.py     # Load JSON results into DataFrames\n"
        "|   |   +-- comparison_tables.py  # Summary statistics tables\n"
        "|   |   +-- plots.py              # Matplotlib / seaborn visualisations\n"
        "|   |   +-- statistical_tests.py  # Welch t-test, Bonferroni, Cohen's d\n"
        "|   +-- utils/\n"
        "|       +-- config.py             # YAML config loader\n"
        "|       +-- metrics.py            # Financial performance metrics\n"
        "+-- scripts/\n"
        "|   +-- download_data.py          # One-time data download CLI\n"
        "|   +-- run_experiments.py        # Main experiment CLI\n"
        "|   +-- launch_experiment.py      # Detached background process launcher\n"
        "|   +-- analyze_results.py        # Post-experiment analysis CLI\n"
        "+-- configs/\n"
        "|   +-- experiment_config.yaml    # 6 experiments, hypotheses, metrics\n"
        "|   +-- training_config.yaml      # Default hyperparameters\n"
        "|   +-- env_config.yaml           # Environment parameters\n"
        "|   +-- data_config.yaml          # Asset list, date ranges\n"
        "+-- results/                      # All experiment outputs (git-ignored)\n"
        "|   +-- experiments/{exp}/{algo}/{strategy}/{run_id}/\n"
        "|       +-- training_metrics.json\n"
        "|       +-- val_results.json      # 2020-2021 validation\n"
        "|       +-- test_results.json     # 2022-2023 test (PRIMARY)\n"
        "|       +-- val_regime_results.json\n"
        "|       +-- test_regime_results.json\n"
        "|       +-- model/\n"
        "+-- data/raw/                     # Downloaded CSV cache (git-ignored)\n"
        "|   +-- training/{ticker}.csv\n"
        "|   +-- validation/{ticker}.csv\n"
        "|   +-- testing/{ticker}.csv\n"
        "+-- requirements.txt\n"
        "+-- CLAUDE.md\n"
    )

    pdf.subsection("Complete File Inventory")
    pdf.table(
        ["File", "Lines", "Role", "Status"],
        [
            ["src/agents/__init__.py",          "~30",  "AGENTS dict, exports",                "Complete"],
            ["src/agents/base_agent.py",         "867",  "BaseAgent, TrainingConfig, Wrapper",  "Complete"],
            ["src/agents/ppo_agent.py",          "780",  "PPO + RegimeObservationWrapper",      "Complete"],
            ["src/agents/sac_agent.py",          "872",  "SAC + RegimeAwareBufferWrapper",      "Complete"],
            ["src/agents/td3_agent.py",          "580",  "TD3 + buffer integration",            "Complete"],
            ["src/agents/rainbow_agent.py",      "1132", "Rainbow DQN (custom loop)",           "Complete"],
            ["src/data/downloader.py",           "926",  "Yahoo Finance + CSV cache",           "Complete"],
            ["src/data/features.py",             "1042", "19 technical indicators",             "Complete"],
            ["src/environment/trading_env.py",   "~400", "Gymnasium portfolio env",             "Complete"],
            ["src/regime/__init__.py",           "337",  "UnifiedRegimeDetector, MarketRegime", "Complete"],
            ["src/regime/rule_based_detector.py","~300", "SMA + volatility rules",              "Complete"],
            ["src/regime/hmm_detector.py",       "~200", "Hidden Markov Model",                 "Complete"],
            ["src/replay/standard_buffer.py",    "~150", "Baseline uniform replay",             "Complete"],
            ["src/replay/regime_aware_buffer.py","~400", "NOVEL CONTRIBUTION",                  "Complete"],
            ["src/training/run_config.py",       "200",  "RunConfig + 6 generators",            "Complete"],
            ["src/training/agent_factory.py",    "~150", "create_agent() factory",              "Complete"],
            ["src/training/experiment_runner.py","700+", "Orchestrator + 3-split pipeline",     "Complete"],
            ["src/training/baselines.py",        "~150", "Buy-hold, 60/40, momentum",           "Complete"],
            ["src/analysis/results_loader.py",   "~200", "JSON -> DataFrame loader",            "Complete"],
            ["src/analysis/comparison_tables.py","~200", "Summary stats tables",                "Complete"],
            ["src/analysis/plots.py",            "~300", "Learning curves, heatmaps",           "Complete"],
            ["src/analysis/statistical_tests.py","~200", "t-tests, effect sizes",               "Complete"],
            ["src/utils/metrics.py",             "424",  "Sharpe, MDD, Sortino, Calmar",        "Complete"],
            ["src/utils/config.py",              "~100", "YAML config loader",                  "Complete"],
            ["scripts/download_data.py",         "~40",  "Data download CLI",                   "Complete"],
            ["scripts/run_experiments.py",       "~130", "Main experiment CLI",                 "Complete"],
            ["scripts/launch_experiment.py",     "~130", "Detached Windows launcher",           "Complete"],
            ["scripts/analyze_results.py",       "~150", "Analysis CLI",                        "Complete"],
            ["configs/experiment_config.yaml",   "~80",  "Experiment specs & hypotheses",       "Complete"],
            ["configs/training_config.yaml",     "~50",  "Default hyperparameters",             "Complete"],
            ["configs/env_config.yaml",          "~30",  "Environment parameters",              "Complete"],
            ["configs/data_config.yaml",         "~20",  "Asset list, date ranges",             "Complete"],
        ],
        col_widths=[70, 14, 60, 26]
    )

    # ====================================================
    # 3. DATA PIPELINE
    # ====================================================
    pdf.section("Module: Data Pipeline  (src/data/)")

    pdf.subsection("downloader.py  --  Yahoo Finance + CSV Cache")
    pdf.body(
        "Purpose: Download OHLCV (Open, High, Low, Close, Volume) price data for all 5 "
        "assets from Yahoo Finance, cache to disk to avoid repeated network requests, "
        "and load the correct split (training/validation/testing) for each experiment run."
    )
    pdf.h3("Key Functions")
    pdf.table(
        ["Function", "Returns", "Description"],
        [
            ["download_single_ticker(ticker, start, end)", "Optional[DataFrame]",
             "Single yfinance download, returns None on failure"],
            ["download_multiple_tickers(tickers, start, end)", "Dict[str, DataFrame]",
             "Downloads list, skips failed tickers gracefully"],
            ["save_to_cache(data, cache_dir)", "None",
             "Saves each ticker to {cache_dir}/{ticker}.csv"],
            ["load_from_cache(tickers, cache_dir)", "Dict[str, DataFrame]",
             "Loads CSVs with proper datetime index parsing"],
            ["download_ohlcv_data(tickers, start, end, cache_dir)", "Dict[str, DataFrame]",
             "MAIN FUNCTION: cache-check -> download -> save -> return"],
            ["download_project_data(cache_dir)", "Dict[period, Dict]",
             "Convenience: downloads all 3 periods with subdirectories"],
        ],
        col_widths=[72, 32, 66]
    )
    pdf.callout(
        "IMPORTANT: Cache Key Design",
        "The cache is keyed by TICKER NAME only (e.g. data/raw/SPY.csv). If all three "
        "date splits share the same cache_dir, the 2020-2021 and 2022-2023 requests will "
        "return the 2010-2019 file. The experiment runner solves this by passing separate "
        "cache_dir values: data/raw/training/, data/raw/validation/, data/raw/testing/."
    )
    pdf.h3("Cache Validity Logic")
    pdf.code(
        "download_ohlcv_data(tickers, start, end, cache_dir='data/raw/training')\n"
        "  for each ticker:\n"
        "    filepath = cache_dir / f'{ticker}.csv'\n"
        "    if filepath.exists() and age < cache_expiry_days:\n"
        "        load_from_cache(ticker, cache_dir)   # O(1) disk read\n"
        "    else:\n"
        "        download_single_ticker(ticker, start, end)  # yfinance HTTP request\n"
        "        save_to_cache(data, cache_dir)        # write CSV for next time\n"
        "  return merged {ticker: DataFrame}"
    )

    pdf.subsection("features.py  --  19 Technical Indicators")
    pdf.body(
        "Purpose: Transform raw OHLCV data into a rich feature set that captures price "
        "trends, momentum, volatility, volume dynamics, and relative performance. Features "
        "are computed per-ticker and normalised using a 60-day rolling window to keep "
        "them stationary across different market regimes."
    )
    pdf.h3("Complete Feature List (19 per asset × 5 assets = 95 market features)")
    pdf.table(
        ["Category", "Features", "Indicator"],
        [
            ["Returns",          "returns",                          "Simple daily return"],
            ["Momentum",         "rsi_14",                           "Relative Strength Index (14d)"],
            ["Momentum",         "macd, macd_signal, macd_hist",     "Moving Average Convergence Divergence"],
            ["Volatility",       "bb_upper/middle/lower/width/%B",   "Bollinger Bands (20d, 2sigma)"],
            ["Volatility",       "atr_14",                           "Average True Range (14d)"],
            ["Volatility",       "volatility_5d/10d/20d",            "Rolling std of returns (annualised)"],
            ["Volume",           "obv",                              "On-Balance Volume (cumulative)"],
            ["Relative",         "relative_strength_5/10/20d",       "Return vs SPY benchmark"],
            ["Microstructure",   "spread_proxy",                     "(High-Low)/Close bid-ask proxy"],
        ],
        col_widths=[28, 60, 82]
    )
    pdf.h3("Main Function")
    pdf.code(
        "calculate_all_features(\n"
        "    data: Dict[str, DataFrame],   # from download_ohlcv_data()\n"
        "    benchmark_ticker='SPY',\n"
        "    normalise=True,               # rolling z-score by default\n"
        "    normalisation_window=60       # 60-day lookback\n"
        ") -> Dict[str, DataFrame]         # same structure, features instead of OHLCV"
    )
    pdf.body(
        "Normalisation: Rolling z-score (rolling mean and std over 60 days) is applied "
        "to all features except raw returns and BB%B (which are already bounded). This "
        "ensures the model receives inputs with consistent scale regardless of whether it "
        "is processing a low-volatility 2017 environment or a high-volatility 2020 environment."
    )

    # ====================================================
    # 4. TRADING ENVIRONMENT
    # ====================================================
    pdf.section("Module: Trading Environment  (src/environment/)")

    pdf.subsection("trading_env.py  --  Gymnasium Portfolio Environment")
    pdf.body(
        "Purpose: A fully Gymnasium-compatible environment that simulates trading 5 assets "
        "over rolling 252-day episodes. The agent submits a continuous allocation vector "
        "which is converted to portfolio weights via softmax, then the environment steps "
        "forward one day at a time applying the portfolio return, transaction costs, "
        "and drawdown penalties."
    )
    pdf.h3("Environment Specification")
    pdf.table(
        ["Property", "Value", "Notes"],
        [
            ["Observation space", "Box(~101,)", "95 market features + 5 weights + 1 port. value"],
            ["Action space",      "Box(-1, +1)^5", "Raw logits, converted to weights via softmax"],
            ["Episode length",    "252 steps", "One trading year (configurable)"],
            ["Initial capital",   "$1,000,000", "Normalised to 1.0 internally"],
            ["Transaction cost",  "0.1% per trade", "Proportional to trade value"],
            ["Termination",       "Portfolio < 10% initial", "Bankruptcy condition"],
            ["Random start",      "Configurable", "True=train, False=eval (fixed start_idx=0)"],
        ],
        col_widths=[40, 40, 90]
    )
    pdf.h3("Reward Function")
    pdf.code(
        "reward = portfolio_return - cost_penalty - drawdown_penalty\n\n"
        "portfolio_return = (curr_value - prev_value) / prev_value\n"
        "cost_penalty     = lambda_cost * (transaction_cost / prev_value)  # lambda=1.0\n"
        "drawdown_penalty = lambda_risk * max(0, drawdown - threshold)     # lambda=0.1, threshold=0.10\n\n"
        "# transaction_cost = sum(|new_weight - old_weight| * portfolio_value) * 0.001"
    )
    pdf.h3("Key Design Choices")
    pdf.bullet([
        "Actions as logits (not weights directly): prevents constraint enforcement issues",
        "Softmax conversion ensures weights sum to 1 and are non-negative",
        "Random start during training: agent sees diverse market conditions each episode",
        "Fixed start during evaluation: reproducible, comparable measurement",
        "current_step attribute exposed: allows regime cache lookup by step index",
        "price_data attribute exposed: allows regime detector to access underlying prices",
    ])

    # ====================================================
    # 5. REGIME DETECTION
    # ====================================================
    pdf.section("Module: Regime Detection  (src/regime/)")

    pdf.subsection("MarketRegime Enum  (canonical definition)")
    pdf.code(
        "# Defined in src/regime/rule_based_detector.py\n"
        "# Re-exported from src/regime/__init__.py\n"
        "# Imported everywhere via: from src.regime import MarketRegime\n\n"
        "class MarketRegime(Enum):\n"
        "    BULL     = 'bull'       # Rising prices, low volatility\n"
        "    BEAR     = 'bear'       # Falling prices, high volatility\n"
        "    SIDEWAYS = 'sideways'   # No clear trend, moderate volatility"
    )

    pdf.subsection("rule_based_detector.py  --  SMA Crossover + Volatility")
    pdf.body(
        "The simplest and most interpretable detector. Uses two rules applied to SPY "
        "(the benchmark) to classify each trading day into one of three regimes."
    )
    pdf.h3("Detection Rules")
    pdf.code(
        "BULL:     SMA_50 > SMA_200  AND  rolling_volatility < 75th_percentile\n"
        "BEAR:     SMA_50 < SMA_200  AND  rolling_volatility > 75th_percentile\n"
        "SIDEWAYS: everything else (ambiguous trend OR low-signal crossover)\n\n"
        "# SMA_50  = 50-day simple moving average of SPY close price\n"
        "# SMA_200 = 200-day simple moving average of SPY close price\n"
        "# rolling_volatility = 20-day rolling std of daily returns\n"
        "# vol_threshold = 75th percentile of all rolling_volatility values in dataset"
    )
    pdf.h3("Key Methods")
    pdf.table(
        ["Method", "Returns", "Description"],
        [
            ["detect(price_data)", "pd.Series[MarketRegime]",
             "Full regime series for all dates -- O(n) computation"],
            ["detect_current(price_data)", "str",
             "Most recent regime as string ('bull'/'bear'/'sideways')"],
            ["detect_at_index(price_data, idx)", "str",
             "Regime at specific step index -- used by replay buffers"],
            ["get_statistics(price_data)", "Dict",
             "Count and % of each regime in dataset"],
        ],
        col_widths=[62, 38, 70]
    )

    pdf.subsection("hmm_detector.py  --  Hidden Markov Model")
    pdf.body(
        "A statistical approach that learns regime transition patterns from training data. "
        "Unlike rule-based detection, HMM discovers regime structure from data rather than "
        "imposing fixed thresholds. Requires the hmmlearn package.\n\n"
        "Key difference: must be fit() on training data before use. "
        "This is done once at the start of each experiment run, using only training period "
        "data (2010-2019) to prevent look-ahead bias."
    )
    pdf.code(
        "# Only used when run_config.detection_method == 'hmm'\n"
        "detector = HMMRegimeDetector(n_states=3)\n"
        "detector.fit(train_price_data, verbose=False)   # Train on 2010-2019 ONLY\n"
        "regimes = detector.predict(price_data)           # Viterbi decoding\n"
        "probs   = detector.predict_proba(price_data)     # Soft probabilities"
    )

    pdf.subsection("__init__.py  --  UnifiedRegimeDetector")
    pdf.body(
        "A facade class that provides a single interface for both detectors. "
        "experiment_runner.py always creates UnifiedRegimeDetector, never the underlying "
        "classes directly. Falls back gracefully if a detector's dependency is missing."
    )
    pdf.code(
        "# Rule-based (no training needed, always available)\n"
        "detector = UnifiedRegimeDetector(method='rule_based')\n\n"
        "# HMM (requires hmmlearn, requires fitting)\n"
        "detector = UnifiedRegimeDetector(method='hmm')\n"
        "detector.fit(train_price_data)\n\n"
        "# Common interface for both:\n"
        "regimes  = detector.detect(price_data)               # pd.Series\n"
        "current  = detector.detect_current(price_data)        # str\n"
        "at_idx   = detector.detect_at_index(price_data, 42)   # str\n"
        "stats    = detector.get_statistics(price_data)         # Dict"
    )
    pdf.callout(
        "PERFORMANCE: Regime Caching Pattern",
        "Calling detect() on every environment step would be very slow (O(n) each time). "
        "All agents and wrappers cache the full regime series at initialisation: "
        "self._regime_cache = detector.detect(price_data). During training, regime "
        "lookups become O(1): regime = self._regime_cache.iloc[current_step]."
    )

    # ====================================================
    # 6. REPLAY BUFFERS
    # ====================================================
    pdf.section("Module: Replay Buffers  (src/replay/)")

    pdf.subsection("standard_buffer.py  --  Baseline Uniform Replay")
    pdf.body(
        "The baseline replay buffer used by the 'standard' strategy condition. "
        "Stores experiences as (state, action, reward, next_state, done) tuples in a "
        "fixed-size circular buffer and samples uniformly at random. This is what "
        "all published DRL work uses by default and what our novel contribution improves upon."
    )
    pdf.code(
        "Experience = NamedTuple(\n"
        "    state:      np.ndarray,   # observation at time t\n"
        "    action:     np.ndarray,   # action taken\n"
        "    reward:     float,         # reward received\n"
        "    next_state: np.ndarray,   # observation at t+1\n"
        "    done:       bool,          # episode terminated?\n"
        "    info:       Dict           # metadata\n"
        ")\n\n"
        "buffer = StandardReplayBuffer(buffer_size=1_000_000)\n"
        "buffer.add(state, action, reward, next_state, done)  # FIFO circular\n"
        "batch  = buffer.sample(batch_size=256)               # uniform random"
    )

    pdf.subsection("regime_aware_buffer.py  --  THE NOVEL CONTRIBUTION")
    pdf.callout(
        "NOVEL CONTRIBUTION",
        "This file contains the core academic contribution of the project. Instead of a "
        "single buffer with uniform sampling, the RegimeAwareReplayBuffer maintains "
        "SEPARATE storage for each market regime and samples from them intelligently. "
        "This ensures the learning gradient is not dominated by whichever regime was "
        "most common during training."
    )
    pdf.h3("Internal Storage Architecture")
    pdf.code(
        "class RegimeAwareReplayBuffer:\n"
        "    def __init__(self, buffer_size, strategy='balanced'):\n"
        "        # Three separate circular buffers, one per regime\n"
        "        self._regime_buffers = {\n"
        "            MarketRegime.BULL:     deque(maxlen=buffer_size // 3),\n"
        "            MarketRegime.BEAR:     deque(maxlen=buffer_size // 3),\n"
        "            MarketRegime.SIDEWAYS: deque(maxlen=buffer_size // 3),\n"
        "        }\n"
        "        self.strategy = SamplingStrategy(strategy)"
    )
    pdf.h3("Sampling Strategy Implementations")
    pdf.code(
        "def _sample_balanced(self, batch_size):\n"
        "    # Take batch_size // 3 from each regime regardless of how full each is\n"
        "    # Ensures equal gradient contribution from BULL, BEAR, SIDEWAYS\n\n"
        "def _sample_current_focused(self, batch_size, current_regime):\n"
        "    # 60% of batch from current_regime, 20% each from others\n"
        "    # Agent focuses on what's happening RIGHT NOW\n\n"
        "def _sample_inverse_frequency(self, batch_size):\n"
        "    # P(regime) prop. 1/count(regime)  ->  rare regimes get more samples\n"
        "    # If BEAR has 10% of data, it gets ~90% of sampling probability\n\n"
        "def _sample_adaptive(self, batch_size):\n"
        "    # P(regime) prop. mean(TD_error in that regime)\n"
        "    # Directs learning toward whichever regime the agent is struggling with"
    )
    pdf.h3("RegimeExperience -- Extended Experience Tuple")
    pdf.code(
        "# Same as Experience but with two extra fields:\n"
        "regime:   MarketRegime   # which regime this experience came from\n"
        "td_error: float          # current TD error (updated by agent after each update)"
    )

    # ====================================================
    # 7. AGENTS
    # ====================================================
    pdf.section("Module: Agents  (src/agents/)")

    pdf.subsection("base_agent.py  --  Unified Interface")
    pdf.body(
        "Defines the abstract base class that all 4 agents implement, plus supporting "
        "dataclasses and the SB3-compatibility wrapper for replay buffers."
    )
    pdf.h3("TrainingConfig  (dataclass)")
    pdf.table(
        ["Field", "Default", "Description"],
        [
            ["total_timesteps",  "1,000,000", "Total environment steps"],
            ["eval_freq",        "50,000",    "Evaluate every N steps"],
            ["n_eval_episodes",  "10",        "Episodes per evaluation"],
            ["save_freq",        "total_ts",  "Save checkpoint every N steps"],
            ["early_stopping",   "False",     "Stop if no improvement"],
            ["patience",         "10",        "Evals without improvement before stop"],
            ["seed",             "42",        "Random seed for reproducibility"],
            ["device",           "'auto'",    "CUDA if available, else CPU"],
            ["tensorboard_log",  "None",      "Tensorboard logging directory"],
        ],
        col_widths=[42, 28, 100]
    )
    pdf.h3("TrainingMetrics  (dataclass) -- What Gets Saved")
    pdf.table(
        ["Field", "Type", "Description"],
        [
            ["timesteps",       "List[int]",         "Evaluation step indices"],
            ["episode_rewards", "List[float]",       "Mean reward per episode"],
            ["eval_rewards",    "List[float]",       "Rewards at eval checkpoints"],
            ["sharpe_ratios",   "List[float]",       "Sharpe ratio trajectory"],
            ["max_drawdowns",   "List[float]",       "Max drawdown trajectory"],
            ["policy_losses",   "List[float]",       "Policy network loss"],
            ["value_losses",    "List[float]",       "Value network loss"],
            ["regime_rewards",  "Dict[str,List]",    "Rewards per regime (bull/bear/sideways)"],
        ],
        col_widths=[42, 32, 96]
    )
    pdf.h3("ReplayBufferWrapper -- SB3 Compatibility Layer")
    pdf.body(
        "Stable Baselines 3 (SB3) has its own replay buffer interface. "
        "ReplayBufferWrapper makes RegimeAwareReplayBuffer look like an SB3 buffer "
        "so it can be swapped in transparently."
    )
    pdf.code(
        "class ReplayBufferWrapper:\n"
        "    \"\"\"Adapts RegimeAwareReplayBuffer to SB3's interface.\"\"\"\n\n"
        "    def add(self, obs, next_obs, action, reward, done, infos):\n"
        "        # 1. Extract regime via fallback chain:\n"
        "        #    info dict -> cached series + env.current_step -> stored regime -> 'sideways'\n"
        "        # 2. Convert to RegimeExperience and add to _regime_buffers[regime]\n\n"
        "    def sample(self, batch_size, env=None):\n"
        "        # 1. Call regime_buffer.sample() with intelligent strategy\n"
        "        # 2. Convert numpy arrays to torch tensors\n"
        "        # 3. Return SB3 ReplayBufferSamples namedtuple"
    )

    pdf.subsection("ppo_agent.py  --  Proximal Policy Optimisation")
    pdf.callout(
        "WHY PPO CANNOT USE THE REPLAY BUFFER DIRECTLY",
        "PPO is an ON-POLICY algorithm: it can only learn from experiences collected "
        "with its CURRENT policy. It discards old experiences after each update batch. "
        "This means a replay buffer is impossible. Regime awareness is instead achieved "
        "by AUGMENTING THE OBSERVATION: the agent sees market features + regime one-hot "
        "encoding, so it learns conditional policies like 'if bear, do X'."
    )
    pdf.h3("RegimeObservationWrapper  (gym.ObservationWrapper)")
    pdf.code(
        "# Extends observation from 98 dims to 101 dims\n"
        "original_obs  = [95 market features, 5 portfolio weights, 1 port. value]  # 101\n"
        "augmented_obs = [original_obs..., bull?, bear?, sideways?]                # 104\n\n"
        "# Wait -- actually the CLAUDE.md says 98 original -> 101 with regime\n"
        "# The wrapper appends [is_bull, is_bear, is_sideways] one-hot\n\n"
        "class RegimeObservationWrapper(gym.ObservationWrapper):\n"
        "    def __init__(self, env, detector, price_data):\n"
        "        self._regime_cache = detector.detect(price_data)   # cache once\n"
        "    def observation(self, obs):\n"
        "        regime = self._regime_cache.iloc[self.env.current_step]\n"
        "        one_hot = [regime==BULL, regime==BEAR, regime==SIDEWAYS]\n"
        "        return np.concatenate([obs, one_hot])"
    )

    pdf.subsection("sac_agent.py  --  Soft Actor-Critic")
    pdf.body(
        "SAC is off-policy and maximises entropy alongside return, encouraging exploration. "
        "It is the primary agent for demonstrating the novel contribution because it directly "
        "integrates RegimeAwareReplayBuffer into the SB3 training loop."
    )
    pdf.h3("Key Hyperparameters")
    pdf.table(
        ["Parameter", "Default", "Notes"],
        [
            ["learning_rate", "3e-4", "Adam optimiser step size"],
            ["buffer_size",   "1M",   "Total replay memory"],
            ["batch_size",    "256",  "Samples per gradient update"],
            ["gamma",         "0.99", "Discount factor (long-term focus)"],
            ["tau",           "0.005","Soft target network update rate"],
            ["ent_coef",      "auto", "Entropy temperature (auto-tuned)"],
            ["learning_starts","10k", "Steps before first update"],
        ],
        col_widths=[42, 28, 100]
    )
    pdf.h3("RegimeTrackingCallback -- Updates Buffer During Training")
    pdf.code(
        "class RegimeTrackingCallback(BaseCallback):\n"
        "    \"\"\"Called every step. Detects current regime, updates the wrapper.\"\"\"\n"
        "    def _on_step(self):\n"
        "        regime_str = env_info.get('regime') or detector.detect_current(price_data)\n"
        "        self.buffer_wrapper.set_current_regime(regime_str)\n"
        "        # Buffer uses this for current_focused strategy"
    )

    pdf.subsection("td3_agent.py  --  Twin Delayed DDPG")
    pdf.body(
        "TD3 addresses the overestimation bias of DDPG by using two critic networks "
        "and taking the minimum, and by delaying policy updates relative to value updates. "
        "Shares the same ReplayBufferWrapper approach as SAC."
    )
    pdf.h3("TD3-Specific Parameters")
    pdf.table(
        ["Parameter", "Default", "Purpose"],
        [
            ["policy_delay",        "2",  "Update actor every 2 critic updates"],
            ["target_policy_noise", "0.2","Noise added to target actions"],
            ["target_noise_clip",   "0.5","Clips target noise to ±0.5"],
            ["action_noise",        "OrnsteinUhlenbeck", "Exploration noise type"],
        ],
        col_widths=[50, 30, 90]
    )

    pdf.subsection("rainbow_agent.py  --  Rainbow DQN")
    pdf.body(
        "Rainbow combines 6 improvements to DQN into a single agent. Because Rainbow "
        "uses a discrete action space, the trading environment must be wrapped with "
        "DiscreteActionWrapper before training."
    )
    pdf.h3("Six Rainbow Components")
    pdf.table(
        ["Component", "What It Does", "Benefit"],
        [
            ["Double DQN", "Separate online/target network for action selection/eval",
             "Reduces overestimation bias"],
            ["Prioritised Replay (PER)", "Sample by TD error priority",
             "Focus on surprising experiences"],
            ["Dueling Networks", "Separate streams for V(s) and A(s,a)",
             "Better value estimation"],
            ["Multi-step returns", "n=3 step return targets",
             "Better credit assignment"],
            ["C51 (Categorical)", "Learn distribution of returns, not just mean",
             "Risk-awareness built in"],
            ["Noisy Networks", "Learned exploration via noisy weights",
             "No need for epsilon-greedy"],
        ],
        col_widths=[42, 70, 58]
    )
    pdf.h3("DiscreteActionWrapper")
    pdf.code(
        "# Converts continuous portfolio space to discrete actions\n"
        "# 3 choices per asset: DECREASE (-10%), HOLD (0%), INCREASE (+10%)\n"
        "# Total action space: 3^5 = 243 combinations\n\n"
        "class DiscreteActionWrapper(gym.ActionWrapper):\n"
        "    def action(self, discrete_action: int) -> np.ndarray:\n"
        "        # Maps integer 0..242 -> (delta_SPY, delta_QQQ, ...) vector\n"
        "        # Then applies softmax to produce valid portfolio weights"
    )
    pdf.h3("Custom Training Loop  (not SB3)")
    pdf.body(
        "Unlike PPO/SAC/TD3, Rainbow does NOT use Stable Baselines 3. It has a "
        "custom training loop that manually handles experience collection, PER sampling, "
        "distributional loss computation, target network updates, and noisy network resets."
    )

    pdf.subsection("agents/__init__.py  --  Registry")
    pdf.code(
        "from src.agents import AGENTS, REPLAY_BUFFER_AGENTS\n\n"
        "AGENTS = {'ppo': PPOAgent, 'sac': SACAgent, 'td3': TD3Agent, 'rainbow': RainbowAgent}\n"
        "REPLAY_BUFFER_AGENTS = ['sac', 'td3', 'rainbow']  # can use replay buffers\n\n"
        "# Quick access:\n"
        "agent = AGENTS['sac'](env, regime_detector=detector, replay_buffer=buffer)"
    )

    # ====================================================
    # 8. TRAINING INFRASTRUCTURE
    # ====================================================
    pdf.section("Module: Training Infrastructure  (src/training/)")

    pdf.subsection("run_config.py  --  The Atomic Unit of Work")
    pdf.body(
        "RunConfig is the single dataclass that describes one complete experiment run. "
        "Every field needed to reproduce a run is stored here. The run_id property "
        "derives a unique string identifier for file naming and manifest tracking."
    )
    pdf.code(
        "@dataclass\n"
        "class RunConfig:\n"
        "    experiment:       str   # 'replay_comparison', 'robustness', etc.\n"
        "    algorithm:        str   # 'ppo', 'sac', 'td3', 'rainbow'\n"
        "    strategy:         str   # 'standard', 'balanced', 'current_focused', etc.\n"
        "    seed:             int   # 42, 123, 456, 789, 1000\n"
        "    total_timesteps:  int   # 500_000 for E1-E5, 1_000_000 for E6\n"
        "\n"
        "    # Experiment-specific (default = inactive)\n"
        "    noise_level:      float = 0.0          # E3: 0.1/0.2/0.3 noise\n"
        "    feature_set:      str   = 'full'       # E4: 'full' or 'reduced'\n"
        "    detection_method: str   = 'rule_based' # E5: 'rule_based' or 'hmm'\n"
        "    eval_checkpoints: List  = None         # E6: [50k, 100k, 250k, 500k, 1M]\n"
        "\n"
        "    @property run_id -> str       # experiment__algo__strategy__seed{N}\n"
        "    @property output_dir -> Path  # results/experiments/{exp}/{algo}/{strat}/{run_id}/"
    )
    pdf.h3("Six Experiment Generators")
    pdf.table(
        ["Function", "Runs", "Variables", "Description"],
        [
            ["generate_e1_runs()", "100", "4 algos × 5 strategies × 5 seeds",
             "Main replay comparison"],
            ["generate_e2_runs()", "60",  "4 algos × 3 strategies × 5 seeds",
             "Per-regime performance"],
            ["generate_e3_runs()", "36",  "4 noise × 3 strategies × 3 seeds (SAC only)",
             "Robustness to noisy labels"],
            ["generate_e4_runs()", "20",  "2 feature sets × 2 algos × 5 seeds",
             "Feature ablation"],
            ["generate_e5_runs()", "12",  "2 methods × 2 strategies × 3 seeds (SAC)",
             "HMM vs rule-based"],
            ["generate_e6_runs()", "30",  "3 strategies × 2 algos × 5 seeds",
             "Sample efficiency (1M steps)"],
        ],
        col_widths=[46, 14, 64, 46]
    )

    pdf.subsection("agent_factory.py  --  create_agent()")
    pdf.body(
        "Takes a RunConfig and builds a fully-configured agent with the correct "
        "replay buffer (or observation wrapper for PPO) attached."
    )
    pdf.code(
        "def create_agent(run_config: RunConfig, train_env, regime_detector) -> BaseAgent:\n"
        "\n"
        "    if run_config.algorithm == 'ppo':\n"
        "        config = PPOConfig(add_regime_to_obs=(strategy != 'standard'))\n"
        "        return PPOAgent(train_env, ppo_config=config, regime_detector=detector)\n"
        "\n"
        "    elif run_config.algorithm == 'sac':\n"
        "        if strategy != 'standard':\n"
        "            buffer = RegimeAwareReplayBuffer(strategy=strategy)\n"
        "            wrapped = RegimeAwareReplayBufferWrapper(buffer, ...)\n"
        "        return SACAgent(train_env, replay_buffer=wrapped, ...)\n"
        "\n"
        "    elif run_config.algorithm == 'td3':\n"
        "        if strategy != 'standard':\n"
        "            buffer = RegimeAwareReplayBuffer(strategy=strategy)\n"
        "        return TD3Agent(train_env, replay_buffer=buffer, ...)\n"
        "\n"
        "    elif run_config.algorithm == 'rainbow':\n"
        "        env = DiscreteActionWrapper(train_env)\n"
        "        buffer = RegimeAwareReplayBuffer(strategy=strategy)\n"
        "        return RainbowAgent(env, replay_buffer=buffer, ...)"
    )

    pdf.subsection("experiment_runner.py  --  Orchestrator")
    pdf.body(
        "The central execution engine. execute_single_run() is the top-level function "
        "that performs one complete training run: data loading, environment creation, "
        "regime detection, training, validation evaluation, and test evaluation."
    )
    pdf.h3("Three-Split Data Pipeline (critical design decision)")
    pdf.code(
        "def execute_single_run(run_config_dict: dict) -> dict:\n"
        "\n"
        "    # 1. Load THREE separate data splits with per-split cache directories\n"
        "    train_data = download_ohlcv_data(tickers, '2010-01-01', '2019-12-31',\n"
        "                                     cache_dir='data/raw/training')\n"
        "    val_data   = download_ohlcv_data(tickers, '2020-01-01', '2021-12-31',\n"
        "                                     cache_dir='data/raw/validation')\n"
        "    test_data  = download_ohlcv_data(tickers, '2022-01-01', '2023-12-31',\n"
        "                                     cache_dir='data/raw/testing')\n"
        "\n"
        "    # 2. Features for each split\n"
        "    train_features = calculate_all_features(train_data)\n"
        "    val_features   = calculate_all_features(val_data)\n"
        "    test_features  = calculate_all_features(test_data)\n"
        "\n"
        "    # 3. Three environments\n"
        "    train_env = TradingEnvironment(train_data, train_features, random_start=True)\n"
        "    val_env   = TradingEnvironment(val_data,   val_features,   random_start=False)\n"
        "    test_env  = TradingEnvironment(test_data,  test_features,  random_start=False)\n"
        "\n"
        "    # 4. Regime detector -- fit on TRAINING DATA ONLY\n"
        "    detector = UnifiedRegimeDetector(method=run_config.detection_method)\n"
        "    detector.fit(train_data)   # HMM only; rule-based never needs fitting\n"
        "\n"
        "    # 5. Agent-specific wrappers on val/test envs\n"
        "    if algorithm == 'rainbow': val_env = DiscreteActionWrapper(val_env); ...\n"
        "    if algorithm == 'ppo' and not standard:\n"
        "        val_env  = RegimeObservationWrapper(val_env,  detector, val_data)\n"
        "        test_env = RegimeObservationWrapper(test_env, detector, test_data)\n"
        "\n"
        "    # 6. Create agent (uses train_detector which may have noise for E3)\n"
        "    agent = create_agent(run_config, train_env, train_detector)\n"
        "\n"
        "    # 7. Train on training data only\n"
        "    metrics = agent.train(training_config=create_training_config(run_config))\n"
        "\n"
        "    # 8. Validate on 2020-2021 -> val_results.json\n"
        "    val_results = agent.evaluate(val_env, n_episodes=10)\n"
        "\n"
        "    # 9. Test on 2022-2023 -> test_results.json (PRIMARY metric)\n"
        "    test_results = agent.evaluate(test_env, n_episodes=10)\n"
        "\n"
        "    # 10. Save all JSON files + model checkpoint\n"
        "    return {sharpe_ratio: test_results, val_sharpe_ratio: val_results, ...}"
    )
    pdf.h3("Manifest -- Resumability")
    pdf.body(
        "Every run's status is tracked in results/experiments/manifest.json. "
        "If the process is killed mid-run, relaunching will skip completed runs "
        "and only execute pending ones. The manifest supports parallel workers "
        "via ProcessPoolExecutor."
    )
    pdf.code(
        "manifest.json structure:\n"
        "{\n"
        "  'last_updated': '2026-02-25T18:10:13',\n"
        "  'summary': {'pending': 98, 'running': 1, 'completed': 1, 'failed': 0},\n"
        "  'runs': [\n"
        "    {\n"
        "      'run_id':  'replay_comparison__ppo__standard__seed42',\n"
        "      'status':  'completed',\n"
        "      'wall_time_seconds': 1449.1,\n"
        "      'result_dir': 'results/experiments/replay_comparison/ppo/standard/...'\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}"
    )

    pdf.subsection("baselines.py  --  Non-RL Comparison Strategies")
    pdf.body(
        "Provides four traditional investment strategies to serve as sanity checks. "
        "A well-trained RL agent should outperform all of these on the test set."
    )
    pdf.table(
        ["Strategy", "Allocation Rule", "Rebalancing"],
        [
            ["Buy & Hold", "100% SPY, no changes", "Never"],
            ["Equal Weight", "20% in each of 5 assets", "Monthly"],
            ["60/40", "60% SPY + 40% TLT", "Monthly"],
            ["Momentum", "Top 2 assets by 12-1 month return", "Monthly"],
        ],
        col_widths=[40, 80, 50]
    )

    # ====================================================
    # 9. ANALYSIS
    # ====================================================
    pdf.section("Module: Analysis  (src/analysis/)")

    pdf.subsection("results_loader.py  --  JSON -> DataFrame")
    pdf.body(
        "Walks the results/experiments/ directory tree, reads all JSON files, "
        "and returns a flat DataFrame with one row per completed run. This is the "
        "input to all other analysis functions."
    )
    pdf.code(
        "loader = ResultsLoader('results/experiments')\n"
        "df = loader.load_all()   # returns pd.DataFrame\n\n"
        "# Columns include:\n"
        "# run_id, experiment, algorithm, strategy, seed,\n"
        "# sharpe_ratio (test), max_drawdown (test),\n"
        "# val_sharpe_ratio, val_max_drawdown,\n"
        "# bull_sharpe, bear_sharpe, sideways_sharpe,  <- from regime eval\n"
        "# wall_time_seconds, ..."
    )

    pdf.subsection("comparison_tables.py  --  Summary Statistics")
    pdf.body(
        "Generates publication-ready tables comparing algorithms and strategies "
        "across all seeds. Tables are exported as both CSV and pretty-printed text."
    )

    pdf.subsection("plots.py  --  Visualisations")
    pdf.body("Generates all figures used in the report:")
    pdf.bullet([
        "Learning curves: mean ± std reward vs timesteps per strategy",
        "Box plots: Sharpe ratio distribution per strategy / algorithm",
        "Heatmap: algorithm vs strategy mean Sharpe (2D grid)",
        "Regime performance radar: bull/bear/sideways Sharpe per strategy",
        "Drawdown comparison bar chart",
        "Sample efficiency curves (E6)",
    ])

    pdf.subsection("statistical_tests.py  --  Hypothesis Testing")
    pdf.body(
        "Implements the statistical tests needed to claim results are significant, "
        "not just noise from random seeds."
    )
    pdf.table(
        ["Test", "When Used", "Output"],
        [
            ["Welch's t-test", "Pairwise strategy comparison", "p-value, t-statistic"],
            ["Bonferroni correction", "Multiple comparisons (5 strategies, 4 algos)", "Adjusted p-value"],
            ["Cohen's d", "Effect size measurement", "Small/medium/large"],
            ["Mann-Whitney U", "Non-parametric alternative", "p-value"],
            ["Friedman test", "Non-parametric across all strategies", "p-value"],
        ],
        col_widths=[45, 65, 60]
    )

    # ====================================================
    # 10. UTILITIES
    # ====================================================
    pdf.section("Module: Utilities  (src/utils/)")

    pdf.subsection("metrics.py  --  Financial Performance Metrics")
    pdf.body("All 9 metrics are calculated by calculate_all_metrics() and saved in test_results.json.")
    pdf.table(
        ["Metric", "Formula", "Good Value"],
        [
            ["Cumulative Return",  "prod(1+r) - 1",                       "> 0%"],
            ["Annualised Return",  "(1+cum_ret)^(252/n) - 1",              "> 5% pa"],
            ["Volatility",         "std(returns) × sqrt(252)",             "Lower = better"],
            ["Sharpe Ratio",       "(ann_ret - rf) / ann_vol",             "> 1.0"],
            ["Sortino Ratio",      "(ann_ret - rf) / downside_std",        "> 1.0"],
            ["Max Drawdown",       "max((peak - trough) / peak)",          "< 20%"],
            ["Calmar Ratio",       "ann_ret / max_drawdown",               "> 0.5"],
            ["Win Rate",           "mean(returns > 0)",                    "> 50%"],
            ["Profit Factor",      "gross_profit / gross_loss",            "> 1.5"],
        ],
        col_widths=[40, 60, 70]
    )

    pdf.subsection("config.py  --  YAML Configuration Loader")
    pdf.body(
        "Loads YAML configuration files from the configs/ directory. "
        "Provides a simple interface to access experiment parameters, "
        "training hyperparameters, and environment settings without "
        "hardcoding values throughout the codebase."
    )

    # ====================================================
    # 11. SCRIPTS
    # ====================================================
    pdf.section("Entry-Point Scripts  (scripts/)")

    pdf.subsection("run_experiments.py  --  Main CLI")
    pdf.code(
        "# Run E1 (all 100 replay comparison runs)\n"
        "python scripts/run_experiments.py --experiments replay_comparison --workers 1\n\n"
        "# Smoke test: all experiments at 10K steps\n"
        "python scripts/run_experiments.py --experiments replay_comparison --timesteps 10000\n\n"
        "# Check progress without running\n"
        "python scripts/run_experiments.py --status\n\n"
        "# Preview what would run\n"
        "python scripts/run_experiments.py --experiments replay_comparison --dry-run"
    )
    pdf.body("Flags: --experiments (list), --workers (int), --timesteps (override), "
             "--dry-run, --status, --retry-failed")

    pdf.subsection("launch_experiment.py  --  Windows Detached Launcher")
    pdf.body(
        "Spawns run_experiments.py as a truly detached background process on Windows "
        "using CREATE_NEW_PROCESS_GROUP and a hidden STARTUPINFO window. Writes PID to "
        "results/experiment.pid. Writes stdout/stderr to results/experiment.log."
    )
    pdf.callout(
        "WINDOWS-SPECIFIC NOTE",
        "DETACHED_PROCESS flag must NOT be used -- it strips the console context and "
        "crashes Intel MKL (used by NumPy) with 'forrtl: error (200)'. Instead, use "
        "CREATE_NEW_PROCESS_GROUP + SW_HIDE via STARTUPINFO. The _is_pid_alive() "
        "helper uses ctypes.windll.kernel32 because os.kill(pid, 0) raises WinError on Windows."
    )
    pdf.code(
        "# Launch E1 in background\n"
        "DRL/Scripts/python.exe scripts/launch_experiment.py --experiments replay_comparison\n\n"
        "# Check if still running + manifest counts + last 20 log lines\n"
        "DRL/Scripts/python.exe scripts/launch_experiment.py --status\n\n"
        "# Kill the process\n"
        "DRL/Scripts/python.exe scripts/launch_experiment.py --kill"
    )

    pdf.subsection("analyze_results.py  --  Post-Experiment Analysis")
    pdf.code(
        "# Full analysis pipeline\n"
        "python scripts/analyze_results.py\n\n"
        "# Output produced in results/analysis/:\n"
        "#   tables/   - CSV summary tables\n"
        "#   plots/    - PNG visualisation figures\n"
        "#   stats/    - JSON statistical test results"
    )

    pdf.subsection("download_data.py  --  One-Time Data Download")
    pdf.code(
        "# Download all 3 date splits, cache to data/raw/{training,validation,testing}/\n"
        "python scripts/download_data.py\n\n"
        "# Equivalent to:\n"
        "from src.data.downloader import download_project_data\n"
        "download_project_data()"
    )

    # ====================================================
    # 12. CONFIGURATION FILES
    # ====================================================
    pdf.section("Configuration Files  (configs/)")

    pdf.subsection("experiment_config.yaml")
    pdf.body("Defines the 6 experiments, their variables, fixed parameters, and success criteria. "
             "This is the single source of truth for the experimental design.")
    pdf.code(
        "experiments:\n"
        "  E1_replay_comparison:\n"
        "    algorithms: [ppo, sac, td3, rainbow]\n"
        "    strategies: [standard, balanced, current_focused, inverse_frequency, adaptive]\n"
        "    seeds: [42, 123, 456, 789, 1000]\n"
        "    total_timesteps: 500000\n"
        "    metrics: [sharpe_ratio, max_drawdown, cumulative_return]\n"
        "    hypothesis: H1\n"
        "\n"
        "success_criteria:\n"
        "  primary:   p < 0.05 for >= 3/4 algorithms\n"
        "  secondary: max_drawdown reduced by > 10%\n"
        "  secondary: bear_market_sharpe improved by > 20%"
    )

    pdf.subsection("training_config.yaml")
    pdf.table(
        ["Parameter", "PPO", "SAC", "TD3", "Rainbow"],
        [
            ["learning_rate",  "3e-4",  "3e-4",  "3e-4",   "6.25e-5"],
            ["batch_size",     "64",    "256",   "256",    "32"],
            ["buffer_size",    "N/A",   "1M",    "1M",     "1M"],
            ["gamma",          "0.99",  "0.99",  "0.99",   "0.99"],
            ["n_steps (PPO)",  "2048",  "N/A",   "N/A",    "N/A"],
            ["n_atoms (C51)",  "N/A",   "N/A",   "N/A",    "51"],
            ["n_step (multi)", "N/A",   "N/A",   "N/A",    "3"],
        ],
        col_widths=[45, 30, 30, 30, 35]
    )

    # ====================================================
    # 13. END-TO-END FLOW
    # ====================================================
    pdf.section("End-to-End Data Flow Walkthrough")

    pdf.subsection("Complete Single Run Walkthrough")
    pdf.body("The following traces exactly what happens for one run: "
             "replay_comparison__sac__balanced__seed42")

    pdf.code(
        "STEP 1  scripts/launch_experiment.py\n"
        "        +-- Spawns run_experiments.py as detached Windows process (PID XXXX)\n"
        "            +-- Writes PID to results/experiment.pid\n"
        "            +-- Redirects stdout/stderr to results/experiment.log\n"
        "\n"
        "STEP 2  scripts/run_experiments.py\n"
        "        +-- Creates ExperimentRunner(['replay_comparison'])\n"
        "        +-- Calls generate_e1_runs() -> 100 RunConfig objects\n"
        "        +-- manifest.register_runs(100_configs)  -> manifest.json\n"
        "        +-- Calls _run_sequential(pending_ids)\n"
        "\n"
        "STEP 3  execute_single_run({'experiment':'replay_comparison','algorithm':'sac',...})\n"
        "\n"
        "STEP 4  DATA LOADING\n"
        "        train_data = download_ohlcv_data([SPY,QQQ,TLT,GLD,BIL],\n"
        "                     '2010-01-01','2019-12-31', cache_dir='data/raw/training')\n"
        "        val_data   = download_ohlcv_data(..., '2020-01-01','2021-12-31',\n"
        "                                          cache_dir='data/raw/validation')\n"
        "        test_data  = download_ohlcv_data(..., '2022-01-01','2023-12-31',\n"
        "                                          cache_dir='data/raw/testing')\n"
        "\n"
        "STEP 5  FEATURE ENGINEERING\n"
        "        train_features = calculate_all_features(train_data)\n"
        "        # -> Dict{SPY: DataFrame[2515 rows x 19 cols], ...}\n"
        "\n"
        "STEP 6  ENVIRONMENT CREATION\n"
        "        train_env = TradingEnvironment(train_data, train_features,\n"
        "                    episode_length=252, random_start=True)\n"
        "        val_env   = TradingEnvironment(val_data,  val_features,\n"
        "                    episode_length=252, random_start=False, start_idx=0)\n"
        "        test_env  = TradingEnvironment(test_data, test_features, ...)\n"
        "\n"
        "STEP 7  REGIME DETECTION\n"
        "        detector = UnifiedRegimeDetector(method='rule_based')\n"
        "        # rule_based: no fitting needed\n"
        "        # Internally: SMA50/SMA200 crossover + volatility threshold on SPY\n"
        "\n"
        "STEP 8  AGENT CREATION (agent_factory)\n"
        "        buffer = RegimeAwareReplayBuffer(buffer_size=1M, strategy='balanced')\n"
        "        wrapper = RegimeAwareReplayBufferWrapper(buffer, obs_space, act_space,\n"
        "                  detector, train_data, train_env)\n"
        "        agent = SACAgent(train_env, replay_buffer=wrapper, ...)\n"
        "        agent._setup_model()  -> creates SB3 SAC, replaces buffer\n"
        "\n"
        "STEP 9  TRAINING  (500,000 timesteps)\n"
        "        For each step:\n"
        "          obs = env.reset() or obs from previous step\n"
        "          action = agent.predict(obs)     # SAC policy\n"
        "          next_obs, reward, done, info = env.step(action)\n"
        "          wrapper.add(obs, next_obs, action, reward, done, info)\n"
        "          # wrapper detects regime from info dict or cache\n"
        "          # experience stored in _regime_buffers[regime]\n"
        "          if step >= learning_starts:\n"
        "              batch = wrapper.sample(256)  # BALANCED: 85 bull+85 bear+86 sideways\n"
        "              losses = agent.update(batch) # standard SAC update\n"
        "\n"
        "STEP 10 VALIDATION EVALUATION  (2020-2021)\n"
        "        val_results = agent.evaluate(val_env, n_episodes=10)\n"
        "        val_financial = _compute_financial_metrics(agent, val_env)\n"
        "        # -> sharpe_ratio, max_drawdown, sortino, calmar, win_rate, ...\n"
        "\n"
        "STEP 11 TEST EVALUATION  (2022-2023)  <- PRIMARY METRIC\n"
        "        test_results = agent.evaluate(test_env, n_episodes=10)\n"
        "        test_financial = _compute_financial_metrics(agent, test_env)\n"
        "\n"
        "STEP 12 SAVE OUTPUTS\n"
        "        results/experiments/replay_comparison/sac/balanced/\n"
        "          replay_comparison__sac__balanced__seed42/\n"
        "            training_metrics.json    <- learning curve data\n"
        "            val_results.json         <- 2020-2021 performance\n"
        "            test_results.json        <- 2022-2023 performance (PRIMARY)\n"
        "            val_regime_results.json  <- per-regime breakdown (val)\n"
        "            test_regime_results.json <- per-regime breakdown (test)\n"
        "            run_config.json          <- full reproducibility info\n"
        "            model/                   <- saved weights\n"
        "\n"
        "STEP 13 manifest.mark_completed(run_id, wall_time)\n"
        "        -> manifest.json updated: completed=2 pending=98\n"
        "        -> next run begins..."
    )

    # ====================================================
    # 14. DEPENDENCY GRAPH
    # ====================================================
    pdf.section("File Dependency Graph")

    pdf.subsection("Import Dependency Map")
    pdf.body("Arrows show which file imports from which. Grouped by module.")
    pdf.code(
        "scripts/launch_experiment.py\n"
        "    (no src imports -- system subprocess only)\n"
        "\n"
        "scripts/run_experiments.py\n"
        "    +-- src/training/experiment_runner.py   (ExperimentRunner)\n"
        "    +-- src/training/run_config.py           (EXPERIMENT_GENERATORS)\n"
        "\n"
        "scripts/analyze_results.py\n"
        "    +-- src/analysis/results_loader.py\n"
        "    +-- src/analysis/comparison_tables.py\n"
        "    +-- src/analysis/plots.py\n"
        "    +-- src/analysis/statistical_tests.py\n"
        "\n"
        "src/training/experiment_runner.py\n"
        "    +-- src/training/run_config.py           (RunConfig)\n"
        "    +-- src/training/agent_factory.py        (create_agent)\n"
        "    +-- src/data/downloader.py               (download_ohlcv_data)\n"
        "    +-- src/data/features.py                 (calculate_all_features)\n"
        "    +-- src/environment/trading_env.py       (TradingEnvironment)\n"
        "    +-- src/regime/__init__.py               (UnifiedRegimeDetector)\n"
        "    +-- src/agents/rainbow_agent.py          (DiscreteActionWrapper)\n"
        "    +-- src/agents/ppo_agent.py              (RegimeObservationWrapper)\n"
        "    +-- src/utils/metrics.py                 (calculate_all_metrics)\n"
        "\n"
        "src/training/agent_factory.py\n"
        "    +-- src/agents/ppo_agent.py              (PPOAgent, PPOConfig)\n"
        "    +-- src/agents/sac_agent.py              (SACAgent, SACConfig)\n"
        "    +-- src/agents/td3_agent.py              (TD3Agent)\n"
        "    +-- src/agents/rainbow_agent.py          (RainbowAgent, DiscreteActionWrapper)\n"
        "    +-- src/replay/regime_aware_buffer.py    (RegimeAwareReplayBuffer)\n"
        "    +-- src/agents/base_agent.py             (TrainingConfig)\n"
        "\n"
        "src/agents/ppo_agent.py\n"
        "    +-- src/agents/base_agent.py             (BaseAgent, TrainingMetrics)\n"
        "    +-- src/regime/__init__.py               (MarketRegime)\n"
        "    +-- stable_baselines3                    (PPO, BaseCallback)\n"
        "    +-- gymnasium\n"
        "\n"
        "src/agents/sac_agent.py\n"
        "    +-- src/agents/base_agent.py             (BaseAgent, TrainingMetrics)\n"
        "    +-- src/replay/regime_aware_buffer.py    (RegimeAwareReplayBuffer)\n"
        "    +-- src/regime/__init__.py               (MarketRegime)\n"
        "    +-- stable_baselines3                    (SAC, ReplayBufferSamples)\n"
        "\n"
        "src/agents/td3_agent.py\n"
        "    +-- src/agents/base_agent.py             (BaseAgent, ReplayBufferWrapper)\n"
        "    +-- src/replay/regime_aware_buffer.py    (RegimeAwareReplayBuffer)\n"
        "    +-- stable_baselines3                    (TD3)\n"
        "\n"
        "src/agents/rainbow_agent.py\n"
        "    +-- src/agents/base_agent.py             (BaseAgent)\n"
        "    +-- src/replay/regime_aware_buffer.py    (RegimeAwareReplayBuffer)\n"
        "    +-- src/regime/__init__.py               (MarketRegime)\n"
        "    +-- torch                                (nn, optim)\n"
        "    +-- gymnasium\n"
        "\n"
        "src/replay/regime_aware_buffer.py\n"
        "    +-- src/regime/__init__.py               (MarketRegime)\n"
        "    +-- src/replay/standard_buffer.py        (Experience)\n"
        "\n"
        "src/regime/__init__.py\n"
        "    +-- src/regime/rule_based_detector.py    (RuleBasedRegimeDetector, MarketRegime)\n"
        "    +-- src/regime/hmm_detector.py           (HMMRegimeDetector)  [optional]\n"
        "\n"
        "src/analysis/*.py\n"
        "    +-- src/utils/metrics.py                 (calculate_all_metrics)\n"
        "    +-- src/training/run_config.py           (EXPERIMENT_GENERATORS)\n"
    )

    pdf.subsection("Dependency Layers  (top-down)")
    pdf.code(
        "LAYER 0: External packages\n"
        "  yfinance, pandas, numpy, gymnasium, stable_baselines3, torch\n"
        "\n"
        "LAYER 1: Core data (no internal dependencies)\n"
        "  src/data/downloader.py\n"
        "  src/data/features.py\n"
        "  src/utils/metrics.py\n"
        "  src/utils/config.py\n"
        "\n"
        "LAYER 2: Domain models (depend on Layer 1)\n"
        "  src/regime/rule_based_detector.py    <- uses pandas, numpy\n"
        "  src/regime/hmm_detector.py           <- uses hmmlearn\n"
        "  src/regime/__init__.py               <- imports Layers 2\n"
        "  src/replay/standard_buffer.py        <- pure numpy\n"
        "\n"
        "LAYER 3: Compound domain (depend on Layer 2)\n"
        "  src/replay/regime_aware_buffer.py    <- uses regime/__init__\n"
        "  src/environment/trading_env.py       <- uses data, utils\n"
        "\n"
        "LAYER 4: Agents (depend on Layers 1-3)\n"
        "  src/agents/base_agent.py             <- depends on replay Layers 2-3\n"
        "  src/agents/ppo_agent.py              <- depends on base + regime\n"
        "  src/agents/sac_agent.py              <- depends on base + regime + replay\n"
        "  src/agents/td3_agent.py              <- depends on base + regime + replay\n"
        "  src/agents/rainbow_agent.py          <- depends on base + regime + replay\n"
        "\n"
        "LAYER 5: Training infrastructure (depend on Layers 1-4)\n"
        "  src/training/run_config.py           <- no src deps (pure dataclass)\n"
        "  src/training/agent_factory.py        <- depends on all agents + replay\n"
        "  src/training/experiment_runner.py    <- depends on all Layers 1-5\n"
        "  src/training/baselines.py            <- depends on data + environment\n"
        "\n"
        "LAYER 6: Analysis (depend on Layers 1-5, reads files)\n"
        "  src/analysis/results_loader.py\n"
        "  src/analysis/comparison_tables.py\n"
        "  src/analysis/plots.py\n"
        "  src/analysis/statistical_tests.py\n"
        "\n"
        "LAYER 7: Entry points (depend on Layers 5-6)\n"
        "  scripts/download_data.py\n"
        "  scripts/run_experiments.py\n"
        "  scripts/launch_experiment.py\n"
        "  scripts/analyze_results.py"
    )

    # ====================================================
    # 15. EXPERIMENTAL DESIGN
    # ====================================================
    pdf.section("Experimental Design")

    pdf.subsection("Full Experiment Matrix")
    pdf.table(
        ["ID", "Name", "Runs", "Variables", "Fixed", "Tests"],
        [
            ["E1", "Replay Comparison",   "100", "4 algos × 5 strats × 5 seeds",
             "500K steps, rule-based det.", "H1, H2, H5"],
            ["E2", "Regime Performance",  "60",  "4 algos × 3 strats × 5 seeds",
             "500K steps",                 "H2, H3"],
            ["E3", "Robustness",          "36",  "4 noise lvls × 3 strats × 3 seeds (SAC)",
             "500K steps",                 "H6"],
            ["E4", "Feature Ablation",    "20",  "2 feature sets × 2 algos × 5 seeds",
             "balanced strategy",          "Feature importance"],
            ["E5", "Detection Method",    "12",  "2 methods × 2 strats × 3 seeds (SAC)",
             "500K steps",                 "HMM vs rule-based"],
            ["E6", "Sample Efficiency",   "30",  "3 strats × 2 algos × 5 seeds",
             "1M steps, 5 checkpoints",    "Learning speed"],
        ],
        col_widths=[12, 38, 14, 48, 36, 22]
    )
    pdf.body(f"Total planned runs: 258  |  E1 is in progress now ({datetime.now().strftime('%d %b %Y')})")

    pdf.subsection("Statistical Analysis Plan")
    pdf.bullet([
        "Primary test: Welch's t-test (unequal variance) for pairwise strategy comparisons",
        "Significance threshold: p < 0.05 after Bonferroni correction for 10 pairwise comparisons",
        "Effect size: Cohen's d -- small (<0.2), medium (<0.5), large (>0.8)",
        "Non-parametric check: Mann-Whitney U test for robustness to normality assumption",
        "Consistency: all 4 algorithms must show same direction for H5",
        "Seeds serve as replication: 5 seeds per configuration",
    ])

    pdf.subsection("Output File Structure for One Run")
    pdf.code(
        "results/experiments/\n"
        "  {experiment}/\n"
        "    {algorithm}/\n"
        "      {strategy}/\n"
        "        {run_id}/\n"
        "          training_metrics.json     <- timesteps, rewards, losses per episode\n"
        "          val_results.json          <- 2020-2021: sharpe, mdd, sortino, ...\n"
        "          test_results.json         <- 2022-2023: PRIMARY METRIC\n"
        "          val_regime_results.json   <- val metrics broken down by bull/bear/sideways\n"
        "          test_regime_results.json  <- test metrics broken down by regime\n"
        "          run_config.json           <- full RunConfig for reproducibility\n"
        "          model/                    <- saved model weights (.zip for SB3, .pt for Rainbow)\n"
        "          error.log                 <- only present if run failed\n"
        "  manifest.json                     <- all run statuses (top-level)"
    )

    # ====================================================
    # 16. FINANCIAL METRICS REFERENCE
    # ====================================================
    pdf.section("Financial Metrics Reference")

    pdf.subsection("Sharpe Ratio  (Primary Metric)")
    pdf.body(
        "The Sharpe Ratio is the single most important metric in this project. "
        "It measures risk-adjusted return: how much excess return is earned per unit "
        "of total volatility. It is used as the primary comparison metric across all experiments."
    )
    pdf.code(
        "Sharpe = (Annualised_Return - Risk_Free_Rate) / Annualised_Volatility\n\n"
        "Annualised_Return = (1 + cumulative_return)^(252/n_days) - 1\n"
        "Annualised_Vol    = std(daily_returns) * sqrt(252)\n\n"
        "Interpretation:\n"
        "  < 0  : Losing money\n"
        "  0-1  : Below average\n"
        "  1-2  : Good\n"
        "  2-3  : Very good\n"
        "  > 3  : Exceptional (or overfitting -- be suspicious)"
    )

    pdf.subsection("Maximum Drawdown  (Risk Metric)")
    pdf.code(
        "MDD = max over time of (peak_value - trough_value) / peak_value\n\n"
        "Example: portfolio goes $100 -> $120 -> $90 -> $130\n"
        "  Peak = $120, Trough = $90\n"
        "  MDD = ($120 - $90) / $120 = 25%\n\n"
        "Target: reduce MDD by > 10% vs standard replay (secondary success criterion)"
    )

    pdf.subsection("Bear Market Sharpe  (Regime-Specific)")
    pdf.body(
        "The most important regime-specific metric. Computed by running the agent only "
        "on episodes that start during a BEAR regime (detected in the test period). "
        "Target: improve bear market Sharpe by > 20% vs standard replay (H3)."
    )

    pdf.subsection("Requirements (requirements.txt)")
    pdf.table(
        ["Package", "Version", "Purpose"],
        [
            ["numpy",            ">=1.24", "Numerical arrays, vectorised math"],
            ["pandas",           ">=2.0",  "DataFrames, time-series manipulation"],
            ["gymnasium",        ">=0.29", "RL environment standard interface"],
            ["stable-baselines3",">=2.0",  "PPO, SAC, TD3 implementations"],
            ["torch",            ">=2.0",  "Neural networks for Rainbow DQN"],
            ["yfinance",         ">=0.2",  "Yahoo Finance OHLCV download"],
            ["pandas-ta",        ">=0.3",  "RSI, MACD, Bollinger Bands, ATR, OBV"],
            ["hmmlearn",         ">=0.3",  "Hidden Markov Model (optional)"],
            ["scikit-learn",     ">=1.3",  "Preprocessing, utilities"],
            ["matplotlib",       ">=3.7",  "Static plots"],
            ["seaborn",          ">=0.12", "Statistical visualisation"],
            ["plotly",           ">=5.14", "Interactive plots"],
            ["streamlit",        ">=1.24", "Live training dashboard (planned)"],
            ["fpdf2",            ">=2.7",  "PDF generation (this document)"],
            ["scipy",            ">=1.10", "Statistical tests"],
        ],
        col_widths=[45, 25, 100]
    )


def main():
    pdf = DocPDF()
    build(pdf)
    out = pathlib.Path("results/repository_audit.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    print(f"[OK] PDF generated: {out.absolute()}")
    print(f"     Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
