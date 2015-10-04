#!/bin/bash

exec 200<$0
flock -n 200 || (echo "apenas um. abortando"; exit 1)

sleep 5

echo "rodei $1 !!"
