#!/bin/bash

# Multi-Disease Prediction System Docker Entrypoint

set -e

echo "🏥 Starting Multi-Disease Prediction System..."

# Check if we should run the pipeline first
if [ "$RUN_PIPELINE" = "true" ]; then
    echo "📊 Running data pipeline..."
    python run_pipeline.py --datasets heart_disease diabetes liver_disease
fi

# Check if we should run tests
if [ "$RUN_TESTS" = "true" ]; then
    echo "🧪 Running tests..."
    pytest tests/ -v
fi

# Start Streamlit app
echo "🚀 Starting Streamlit application..."
exec streamlit run app/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false 