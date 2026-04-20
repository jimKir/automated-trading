# ============================================================
#  Paper environment — active paper trading with market-hours schedules
#  EventBridge starts bot 1h before US market open, stops 5min after close (Mon-Fri)
# ============================================================

environment      = "paper"
trading_mode     = "paper"
image_tag        = "paper-latest"
desired_count    = 1                 # keep running — schedules control start/stop
enable_schedules = true              # auto start/stop on market hours

# Resources — needs headroom for Java/H2O AutoML + trading strategy
task_cpu    = "1024"
task_memory = "2048"
