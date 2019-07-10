#!/usr/bash
set -e

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id/);
aws ec2 stop-instances --region us-west-2 --instance-ids $INSTANCE_ID;

exit 0
