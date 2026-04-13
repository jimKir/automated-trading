# ============================================================
#  Paper environment — on-demand strategy testing
#  No EventBridge schedules — start/stop manually:
#    aws ecs update-service --cluster trading-bot-paper-cluster \
#      --service trading-bot-paper-service --desired-count 1 --region eu-north-1
# ============================================================

environment      = "paper"
trading_mode     = "paper"
desired_count    = 0                 # off by default — start manually when testing
enable_schedules = false             # no auto-schedules — fully on-demand

# Resources (can be smaller since it's testing only)
task_cpu    = "512"
task_memory = "1024"
