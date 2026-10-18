#!/bin/sh
# usage: sh worker.sh queue.txt   (run two of these with nohup; each runs the queue's lines one by one)
cd "$(dirname "$0")"
while line=$(python3 pop.py "$1"); do
  echo "$(date +%H:%M:%S) $line"
  sh -c "$line"
done
