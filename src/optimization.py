"""
Promotion Optimization Module — v2.0
Price elasticity, Monte Carlo simulation, portfolio optimization, break-even analysis.
"""

import numpy as np
import pandas as pd
import yaml


class PromotionOptimizer:
    """
    Advanced promotion optimization with elasticity estimation,
    Monte Carlo uncertainty quantification, and multi-product portfolio optimization.
    """

    def __init__(self, model=None, scaler=None, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = model
        self.scaler = scaler

        opt_cfg = self.config.get('optimization', {})
        self.base_price = opt_cfg.get('base_price', 10)
        self.promo_discount = opt_cfg.get('promo_discount', 0.20)
        self.cost = opt_cfg.get('cost_per_unit', 6)
        self.mc_samples = opt_cfg.get('monte_carlo_samples', 1000)
        self.elasticity_default = opt_cfg.get('elasticity_default', -1.5)
        self.look_back = self.config['model'].get('look_back_days', 30)

    # ------------------------------------------------------------------
    # Basic Promo Simulation (LSTM-based, backward compatible)
    # ------------------------------------------------------------------

    def optimize(self, recent_data):
        """
        Compare No-Promo vs Promo using LSTM predictions.
        Backward-compatible with v1.0 interface.
        """
        print("\n--- 🤖 RUNNING PROMOTION SIMULATION ---")

        current_seq = recent_data[-self.look_back:]
        input_seq = current_seq.reshape(1, self.look_back, -1)
        PROMO_IDX = 1

        # Scenario 1: No Promotion
        seq_no = input_seq.copy()
        seq_no[0, -1, PROMO_IDX] = 0
        pred_no = self.model.predict(seq_no, verbose=0) if hasattr(self.model, 'predict') else self.model(seq_no)

        # Scenario 2: With Promotion
        seq_yes = input_seq.copy()
        seq_yes[0, -1, PROMO_IDX] = 1
        pred_yes = self.model.predict(seq_yes, verbose=0) if hasattr(self.model, 'predict') else self.model(seq_yes)

        sales_no = self._inverse_sales(pred_no, current_seq)
        sales_yes = self._inverse_sales(pred_yes, current_seq)

        promo_price = self.base_price * (1 - self.promo_discount)
        profit_no = (self.base_price - self.cost) * sales_no
        profit_yes = (promo_price - self.cost) * sales_yes

        print(f"🔮 No Promo:   {sales_no:.0f} units → ${profit_no:,.2f}")
        print(f"🔥 With Promo: {sales_yes:.0f} units → ${profit_yes:,.2f}")

        if profit_yes > profit_no:
            print(f"✅ RUN PROMOTION (+${profit_yes - profit_no:,.2f})")
        else:
            print(f"❌ DO NOT PROMOTE (would lose ${profit_no - profit_yes:,.2f})")

        return {
            'sales_no_promo': sales_no,
            'sales_promo': sales_yes,
            'profit_no_promo': profit_no,
            'profit_promo': profit_yes,
            'recommendation': 'promote' if profit_yes > profit_no else 'no_promote'
        }

    # ------------------------------------------------------------------
    # Multi-Scenario Price Simulation
    # ------------------------------------------------------------------

    def simulate_scenarios(self, base_sales, scenarios=None):
        """
        Compare profit across multiple pricing scenarios.

        Parameters
        ----------
        base_sales : predicted baseline sales (no promo)
        scenarios : list of dicts with 'name', 'price', 'elasticity'

        Returns
        -------
        DataFrame with scenario comparisons
        """
        if scenarios is None:
            scenarios = [
                {'name': 'No Promotion', 'price': self.base_price, 'elasticity': 0},
                {'name': '10% Off', 'price': self.base_price * 0.90, 'elasticity': self.elasticity_default},
                {'name': '15% Off', 'price': self.base_price * 0.85, 'elasticity': self.elasticity_default},
                {'name': '20% Off', 'price': self.base_price * 0.80, 'elasticity': self.elasticity_default},
                {'name': '25% Off', 'price': self.base_price * 0.75, 'elasticity': self.elasticity_default},
            ]

        results = []
        for s in scenarios:
            pct_change = (s['price'] - self.base_price) / self.base_price
            demand_multiplier = 1 + s['elasticity'] * pct_change if s['elasticity'] != 0 else 1.0
            adjusted_sales = base_sales * demand_multiplier
            profit = (s['price'] - self.cost) * adjusted_sales
            revenue = s['price'] * adjusted_sales

            results.append({
                'scenario': s['name'],
                'price': s['price'],
                'predicted_sales': adjusted_sales,
                'revenue': revenue,
                'profit': profit,
                'profit_uplift_pct': ((profit / ((self.base_price - self.cost) * base_sales)) - 1) * 100
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Price Elasticity Estimation
    # ------------------------------------------------------------------

    def estimate_elasticity(self, df, price_col='onpromotion', sales_col='sales'):
        """
        Estimate price elasticity from historical promo data.
        Uses simple promo on/off comparison.
        """
        if price_col not in df.columns or sales_col not in df.columns:
            return self.elasticity_default

        promo_on = df[df[price_col] == 1][sales_col].mean()
        promo_off = df[df[price_col] == 0][sales_col].mean()

        if promo_off == 0 or np.isnan(promo_on) or np.isnan(promo_off):
            return self.elasticity_default

        # % change in quantity / % change in price
        pct_qty = (promo_on - promo_off) / promo_off
        pct_price = -self.promo_discount  # price drops by discount %

        elasticity = pct_qty / pct_price if pct_price != 0 else self.elasticity_default
        print(f"  Estimated elasticity: {elasticity:.2f}")
        return elasticity

    # ------------------------------------------------------------------
    # Monte Carlo Simulation
    # ------------------------------------------------------------------

    def monte_carlo_profit(self, base_sales, price, cost=None, n_samples=None,
                           sales_std_pct=0.15):
        """
        Monte Carlo simulation for profit uncertainty.

        Parameters
        ----------
        base_sales : predicted sales baseline
        price : selling price
        cost : cost per unit (defaults to config)
        n_samples : number of simulations
        sales_std_pct : standard deviation as % of base_sales

        Returns
        -------
        dict with mean, std, p5, p50, p95 profit estimates
        """
        cost = cost or self.cost
        n = n_samples or self.mc_samples

        # Sample sales from normal distribution
        sales_samples = np.random.normal(base_sales, base_sales * sales_std_pct, n)
        sales_samples = np.maximum(sales_samples, 0)  # No negative sales

        profit_samples = (price - cost) * sales_samples

        return {
            'mean': np.mean(profit_samples),
            'std': np.std(profit_samples),
            'p5': np.percentile(profit_samples, 5),
            'p25': np.percentile(profit_samples, 25),
            'p50': np.percentile(profit_samples, 50),
            'p75': np.percentile(profit_samples, 75),
            'p95': np.percentile(profit_samples, 95),
            'samples': profit_samples
        }

    # ------------------------------------------------------------------
    # Break-Even Calculator
    # ------------------------------------------------------------------

    def break_even_volume(self, price=None, cost=None, fixed_cost=0):
        """Calculate break-even volume at given price."""
        p = price or self.base_price
        c = cost or self.cost
        margin = p - c
        if margin <= 0:
            return float('inf')
        return fixed_cost / margin if fixed_cost > 0 else 0

    def break_even_price(self, volume, cost=None, fixed_cost=0):
        """Calculate break-even price for given volume."""
        c = cost or self.cost
        if volume <= 0:
            return float('inf')
        return c + fixed_cost / volume

    # ------------------------------------------------------------------
    # Historical Promo Effectiveness
    # ------------------------------------------------------------------

    def promo_effectiveness_by_family(self, df, sales_col='sales', promo_col='onpromotion'):
        """
        Compute promo lift per product family.

        Returns
        -------
        DataFrame: family, avg_sales_promo, avg_sales_no_promo, lift_pct
        """
        if 'family' not in df.columns:
            return pd.DataFrame()

        results = []
        for fam in df['family'].unique():
            sub = df[df['family'] == fam]
            promo = sub[sub[promo_col] == 1][sales_col].mean()
            no_promo = sub[sub[promo_col] == 0][sales_col].mean()

            if pd.isna(promo) or pd.isna(no_promo) or no_promo == 0:
                continue

            results.append({
                'family': fam,
                'avg_sales_promo': promo,
                'avg_sales_no_promo': no_promo,
                'lift_pct': ((promo - no_promo) / no_promo) * 100,
                'promo_days': int(sub[promo_col].sum()),
            })

        return pd.DataFrame(results).sort_values('lift_pct', ascending=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inverse_sales(self, pred_value, reference_data):
        """Un-scale predicted sales value."""
        if self.scaler is None:
            return float(pred_value.flatten()[0])
        dummy = np.zeros((1, reference_data.shape[1]))
        dummy[0, 0] = pred_value.flatten()[0]
        return self.scaler.inverse_transform(dummy)[0, 0]