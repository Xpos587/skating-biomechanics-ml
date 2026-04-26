#!/bin/bash
# Monitor heatmap generation progress

ssh vastai "
cd /root/skating-biomechanics-ml/experiments/yolo26-pose-kd

echo \"=== Heatmap Generation Progress ===\"
echo \"\$(date '+%Y-%m-%d %H:%M:%S')\"
echo ''

# Check process
if ps aux | grep -q 'generate_teacher_heatmaps.*batch-size 128' | grep -v grep; then
    echo '✅ Process RUNNING'
else
    echo '❌ Process NOT RUNNING'
    exit 1
fi

# Get progress from log
PROGRESS_LINE=\$(grep 'Generating heatmaps' logs/heatmap_gen.log | tail -1)
echo \"Progress: \$PROGRESS_LINE\"

# Extract numbers
if [[ \$PROGRESS_LINE =~ ([0-9]+)/([0-9]+) ]]; then
    CURRENT=\${BASH_REMATCH[1]}
    TOTAL=\${BASH_REMATCH[2]}
    PCT=\$(echo \"scale=2; 100 * \$CURRENT / \$TOTAL\" | bc)
    echo \"Completed: \$CURRENT / \$TOTAL (\$Pct%)\"

    # Calculate remaining
    REMAINING=\$((TOTAL - CURRENT))
    echo \"Remaining: \$REMAINING\"
fi

# File size
if [ -f data/teacher_heatmaps.h5 ]; then
    SIZE=\$(ls -lh data/teacher_heatmaps.h5 | awk '{print \$5}')
    echo \"File size: \$SIZE\"
fi

# Log file size
LOG_SIZE=\$(ls -lh logs/heatmap_gen.log | awk '{print \$5}')
echo \"Log size: \$LOG_SIZE\"
"
