output "ecr_repository_url" {
  description = "ECR repository URL for Docker push"
  value       = aws_ecr_repository.trading.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.trading.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.trading.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for tailing logs"
  value       = aws_cloudwatch_log_group.trading.name
}

output "environment" {
  description = "Active environment"
  value       = var.environment
}

output "schedules_enabled" {
  description = "Whether EventBridge market-hours schedules are active"
  value       = var.enable_schedules
}

output "start_command" {
  description = "Command to manually start the service"
  value       = "aws ecs update-service --cluster ${aws_ecs_cluster.trading.name} --service ${aws_ecs_service.trading.name} --desired-count 1 --region ${var.aws_region}"
}

output "stop_command" {
  description = "Command to manually stop the service"
  value       = "aws ecs update-service --cluster ${aws_ecs_cluster.trading.name} --service ${aws_ecs_service.trading.name} --desired-count 0 --region ${var.aws_region}"
}

output "logs_command" {
  description = "Command to tail live logs"
  value       = "aws logs tail ${aws_cloudwatch_log_group.trading.name} --follow --region ${var.aws_region}"
}
