import math

class RiskManager:
    def __init__(self, account_balance, risk_per_trade=0.01):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, entry_price, stop_loss, pip_value=10, is_forex=True):
        """
        Calculates position size based on risk parameters.
        pip_value: Value of 1 pip for a standard lot (default $10 for most majors)
        """
        amount_to_risk = self.account_balance * self.risk_per_trade
        
        if entry_price == stop_loss:
            return 0
            
        risk_per_unit = abs(entry_price - stop_loss)
        
        if is_forex:
            # Forex calculation: (Risk Amount) / (Pips at Risk * Pip Value)
            # Assuming 4th decimal for pips (0.0001)
            pips_at_risk = risk_per_unit / 0.0001
            if pips_at_risk == 0: return 0
            lot_size = amount_to_risk / (pips_at_risk * pip_value)
            return round(lot_size, 2)
        else:
            # Crypto/Stock calculation: (Risk Amount) / (Price Difference)
            position_units = amount_to_risk / risk_per_unit
            return round(position_units, 4)

    def calculate_atr_sl_tp(self, entry_price, atr, multiplier_sl=1.5, multiplier_tp=3.0, direction="long"):
        """
        Calculates SL and TP based on ATR.
        """
        if direction == "long":
            sl = entry_price - (atr * multiplier_sl)
            tp = entry_price + (atr * multiplier_tp)
        else:
            sl = entry_price + (atr * multiplier_sl)
            tp = entry_price - (atr * multiplier_tp)
            
        return round(sl, 5), round(tp, 5)

    def get_risk_reward_ratio(self, entry_price, stop_loss, take_profit):
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - take_profit)
        if risk == 0: return 0
        return round(reward / risk, 2)
