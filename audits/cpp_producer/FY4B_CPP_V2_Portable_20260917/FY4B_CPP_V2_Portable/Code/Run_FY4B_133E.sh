#!/bin/bash

# ========== 配置 ==========
# START="2023-07-01T00:00:00"
# END="2023-12-31T23:59:00"
START="2023-01-01T00:00:00"
END="2023-12-31T23:59:00"
FREQ="15min"
PYTHON_SCRIPT="./Retrieval_FY4B_133E.py"
CONDA_ENV="yangzx"
# ==========================

# 从 START/END 自动提取日期部分生成日志名
S_DATE=${START:0:10}   # 2023-07-01
E_DATE=${END:0:10}     # 2023-12-31
LOG_FILE="./Logs/FY4B_133E_${S_DATE}-${E_DATE}.log"

mkdir -p "$(dirname "$LOG_FILE")"
> "$LOG_FILE"

# 生成时间列表
TIME_LIST=$(python3 -c "
import pandas as pd
times = pd.date_range('$START', '$END', freq='$FREQ')
for t in times:
    print(t.strftime('%Y-%m-%dT%H:%M:%S'))
")

TOTAL=$(echo "$TIME_LIST" | wc -l)
CURRENT=0

for RUN_TIME in $TIME_LIST; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Processing: $RUN_TIME" | tee -a "$LOG_FILE"

    conda run -n "$CONDA_ENV" python "$PYTHON_SCRIPT" "$RUN_TIME" >> "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✅ Success" | tee -a "$LOG_FILE"
    else
        echo "  ❌ Failed (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
    fi
done

echo "All done. $TOTAL time steps processed." | tee -a "$LOG_FILE"

