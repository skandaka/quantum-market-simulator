#!/bin/bash

echo "Running Classiq API checks..."
echo "============================="

cd "$(dirname "$0")"

echo -e "\n1. Checking Classiq API availability:"
python check_classiq_api.py

echo -e "\n2. Testing basic Classiq functionality:"
python simple_classiq_test.py

echo -e "\n3. Testing imports:"
python test_imports.py

echo -e "\nCheck complete!"