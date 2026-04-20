# ============================================================
#  Production environment — always active during market hours
#  EventBridge auto-starts at 13:25 UTC (1h pre-market), stops at 21:05 UTC
# ============================================================

environment      = "production"
trading_mode     = "paper"           # still paper trading — switch to "live" when ready
desired_count    = 1                 # running by default (EventBridge manages scaling)
enable_schedules = true              # auto start/stop on market hours

# Resources
task_cpu    = "512"
task_memory = "1024"
