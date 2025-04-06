import logging
from decimal import Decimal
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression

from hummingbot.core.data_type.common import OrderType, TradeType, PriceType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.connector.connector_base import ConnectorBase


class MLVolInventoryMaker(ScriptStrategyBase):
    exchange = "binance_paper_trade"
    trading_pair = "ETH-USDT"
    price_source = PriceType.MidPrice

    markets = {exchange: {trading_pair}}

    order_amount = Decimal("0.01")
    order_refresh_time = 15  # seconds
    bid_spread = Decimal("0.001")
    ask_spread = Decimal("0.001")
    atr_period = 14
    sma_period = 20
    inventory_target = Decimal("0.5")  # 50% base, 50% quote

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.create_timestamp = 0
        self.mid_prices: List[float] = []
        self.model = LinearRegression()
        self.trained = False
        self.starting_base = None
        self.starting_quote = None

    def on_tick(self):
        if not self.ready_to_trade:
            return

        if self.starting_base is None or self.starting_quote is None:
            base, quote = self.trading_pair.split("-")
            self.starting_base = self.connectors[self.exchange].get_balance(base)
            self.starting_quote = self.connectors[self.exchange].get_balance(quote)

        if self.current_timestamp < self.create_timestamp:
            return

        self.cancel_all_orders()

        mid_price_float = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        mid_price = Decimal(str(mid_price_float))
        self.mid_prices.append(float(mid_price))
        if len(self.mid_prices) > self.sma_period:
            self.mid_prices.pop(0)

        # Train model
        if len(self.mid_prices) >= 5:
            X = np.arange(len(self.mid_prices)).reshape(-1, 1)
            y = np.array(self.mid_prices)
            self.model.fit(X, y)
            self.trained = True

        price_shift = Decimal("0")
        if self.trained:
            future_idx = np.array([[len(self.mid_prices)]])
            prediction = self.model.predict(future_idx)[0]
            price_shift = Decimal(str(prediction)) - mid_price

        base_token, quote_token = self.trading_pair.split("-")
        base_balance = Decimal(str(self.connectors[self.exchange].get_balance(base_token)))
        quote_balance = Decimal(str(self.connectors[self.exchange].get_balance(quote_token)))

        current_base_value = base_balance * mid_price
        total_value = current_base_value + quote_balance
        base_ratio = current_base_value / total_value if total_value > 0 else Decimal("0")
        inventory_skew = base_ratio - self.inventory_target

        # Adjust price with proper skew
        ref_price = mid_price + price_shift
        bid_price = ref_price * (Decimal("1") - self.bid_spread + inventory_skew)
        ask_price = ref_price * (Decimal("1") + self.ask_spread + inventory_skew)

        # Sanity check
        if bid_price <= 0 or ask_price <= 0 or bid_price >= ask_price:
            self.log_with_clock(logging.WARNING, f"Invalid prices — Bid: {bid_price}, Ask: {ask_price}, skipping order.")
            return

        # Debug info
        self.log_with_clock(
            logging.INFO,
            f"Prices — Mid: {mid_price:.4f}, Ref: {ref_price:.4f}, "
            f"Shift: {price_shift:.4f}, Skew: {inventory_skew:.4f}, "
            f"Bid: {bid_price:.4f}, Ask: {ask_price:.4f}"
        )

        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=self.order_amount,
            price=bid_price
        )

        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=self.order_amount,
            price=ask_price
        )

        proposal = [buy_order, sell_order]
        adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

        if not adjusted:
            self.log_with_clock(logging.WARNING, "No orders passed the budget check.")
        else:
            for order in adjusted:
                self.log_with_clock(logging.INFO, f"Placing {order.order_side.name} order: {order.amount} @ {order.price}")
                self.place_order(self.exchange, order)

        self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.BUY:
            self.buy(connector_name, order.trading_pair, order.amount, order.order_type, order.price)
        elif order.order_side == TradeType.SELL:
            self.sell(connector_name, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} "
            f"{self.exchange} at {round(event.price, 2)}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        if self.trained:
            lines.append("\n  ML Model trained with recent mid prices.")
        else:
            lines.append("\n  ML Model is still warming up...")

        return "\n".join(lines)
