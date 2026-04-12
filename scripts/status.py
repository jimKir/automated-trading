#!/usr/bin/env python3
"""Paper trading status. Usage: python scripts/status.py [--watch] [--interval 30]"""
import argparse, os, sys, time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true")
parser.add_argument("--interval", type=int, default=30)
args = parser.parse_args()

def load_env():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
load_env()

G = lambda t: f"\033[92m{t}\033[0m"
R = lambda t: f"\033[91m{t}\033[0m"
B = lambda t: f"\033[1m{t}\033[0m"
D = lambda t: f"\033[2m{t}\033[0m"
pnl = lambda v: (G if float(v)>=0 else R)(f"{float(v):+,.2f}")
pct = lambda v: (G if float(v)*100>=0 else R)(f"{float(v)*100:+.2f}%")

def run():
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit(R("No credentials. Set ALPACA_API_KEY + ALPACA_API_SECRET in .env"))
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus as OS
    c = TradingClient(key, sec, paper=True)
    acct = c.get_account(); pos = c.get_all_positions()
    ords = c.get_orders(filter=GetOrdersRequest(status=OS.ALL, limit=5))
    eq = float(acct.equity); le = float(acct.last_equity); td = eq - le
    BAR = "=" * 52
    print(f"\n{B(BAR)}")
    print(f"  {B('PAPER TRADING')}  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{B(BAR)}")
    print(f"\n{B('ACCOUNT')}")
    print(f"  Equity:       ${eq:>12,.2f}")
    print(f"  Cash:         ${float(acct.cash):>12,.2f}")
    print(f"  Buying Power: ${float(acct.buying_power):>12,.2f}")
    print(f"  Today P&L:    {pnl(td):>18}  ({pct(td/le if le else 0)})")
    print(f"  Status:       {G(str(acct.status.value))}")
    print(f"\n{B(f'POSITIONS ({len(pos)})')}")
    if not pos:
        print(f"  {D('none')}")
    else:
        print(f"  {'Symbol':<8} {'Qty':>7} {'Avg':>9} {'Now':>9} {'P&L':>10} {'%':>7}")
        print(f"  {'-'*52}")
        for p in sorted(pos, key=lambda x: abs(float(x.unrealized_pl)), reverse=True):
            print(f"  {p.symbol:<8} {float(p.qty):>7.2f} {float(p.avg_entry_price):>9.2f}"
                  f" {float(p.current_price):>9.2f} {pnl(p.unrealized_pl):>16}"
                  f" {pct(float(p.unrealized_plpc)):>13}")
    print(f"\n{B('ORDERS (last 5)')}")
    if not ords: print(f"  {D('none')}")
    else:
        for o in ords:
            t = o.created_at.strftime("%H:%M:%S") if o.created_at else "—"
            pr = f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "—"
            sd = G("BUY") if str(o.side.value)=="buy" else R("SELL")
            print(f"  {t}  {o.symbol:<7} {sd} {float(o.filled_qty or o.qty or 0):.0f} @ {pr}  {o.status.value}")
    print(f"\n{D(BAR)}\n")

if args.watch:
    try:
        while True:
            os.system("clear"); run(); time.sleep(args.interval)
    except KeyboardInterrupt: print("\nStopped.")
else:
    run()
