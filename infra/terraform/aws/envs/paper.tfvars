# ============================================================
#  Paper environment — on-demand strategy testing
#  No EventBridge schedules — start/stop manually:
#    aws ecs update-service --cluster trading-bot-paper-cluster \
#      --service trading-bot-paper-service --desired-count 1 --region eu-north-1
# ============================================================

environment      = "paper"
trading_mode     = "paper"
image_tag        = "paper-latest"
desired_count    = 0                 # off by default — start manually when testing
enable_schedules = false             # no auto-schedules — fully on-demand

# Resources — needs headroom for Java/H2O AutoML + trading strategy
task_cpu    = "1024"
task_memory = "2048"
