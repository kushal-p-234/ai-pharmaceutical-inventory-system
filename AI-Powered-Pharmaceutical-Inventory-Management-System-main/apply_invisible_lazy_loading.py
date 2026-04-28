import os
import shutil
import re

# 1. Restore app.py from the backup that had lazy loading
shutil.copy('app.py.bak', 'app.py')

# 2. Read app.py to remove the Lite Mode toggle UI
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the Lite Mode Toggle section from sidebar
toggle_regex = r'# Lite Mode Toggle for performance optimization.*?else:\n    os\.environ\["DISABLE_HEAVY_FEATURES"\] = "false"\n'
content = re.sub(toggle_regex, '', content, flags=re.DOTALL)

# Also remove any other DISABLE_HEAVY_FEATURES warnings in app.py
content = re.sub(r'st\.warning\("⚠️ Deep Learning features are disabled.*?"\)', 'pass', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 3. Process deep_learning_forecasting.py
with open('deep_learning_forecasting.py', 'r', encoding='utf-8') as f:
    dl_content = f.read()
# Remove top level tensorflow imports and DISABLE_HEAVY logic
dl_content = re.sub(r'# Conditional TensorFlow import.*?warnings\.warn\("TensorFlow not available - deep learning features disabled"\)', '', dl_content, flags=re.DOTALL)
# It's already removed? Wait, I previously reverted dl_content. Let's just do a clean replace.
dl_content = dl_content.replace('from sklearn.preprocessing import MinMaxScaler', 'from sklearn.preprocessing import MinMaxScaler')

# We'll handle dl_content and others with replace_file_content later if this script is too complex.
print("App.py restored with invisible lazy loading.")
