# Development Guide

This guide explains how to set up the HydraRP library for local development.

## Quick Setup (Editable Install)

Install the package in "editable" or "development" mode so changes are immediately reflected without reinstalling:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/AlexRice13/HydraRP_Neko.git
cd HydraRP_Neko

# Install in editable mode
pip install -e .
```

This creates a link to your development directory, so any changes you make to the code are immediately available when you import the package.

## Verify Installation

Test that the package is installed correctly:

```python
# In Python or Jupyter notebook
from hydra_rp import set_config, legal_reward_fn
print("✓ HydraRP installed successfully!")
```

## Development Workflow

### 1. Make Changes

Edit files in the `hydra_rp/` directory:
```bash
# Example: edit a reward function
nano hydra_rp/rewards.py
```

### 2. Test Immediately

No need to reinstall! Just restart your Python interpreter or Jupyter kernel:

```python
# In Jupyter, restart kernel (Kernel -> Restart)
# Then import again
from hydra_rp import legal_reward_fn

# Your changes are now active!
```

### 3. Test in Jupyter Notebook

```python
# Cell 1: Import (will use your local development version)
from hydra_rp import set_config, legal_reward_fn

# Cell 2: Configure
set_config(
    judge_api_key="test-key",
    judge_model_name="gpt-4"
)

# Cell 3: Test your changes
prompts = [[{"role": "user", "content": "test"}]]
completions = [{"content": "test response"}]
scores = legal_reward_fn(prompts, completions, task_type=["chat"])
print(f"Score: {scores[0]}")
```

### 4. Reload Module (Advanced)

If you need to reload a module without restarting Jupyter:

```python
import importlib
import hydra_rp.rewards
importlib.reload(hydra_rp.rewards)
```

## Alternative: Direct Path Method

If you don't want to install, you can add the repository to your Python path:

### Option A: In Jupyter Notebook

```python
import sys
sys.path.insert(0, '/path/to/HydraRP_Neko')

from hydra_rp import set_config, legal_reward_fn
```

### Option B: PYTHONPATH Environment Variable

```bash
# In terminal
export PYTHONPATH="/path/to/HydraRP_Neko:$PYTHONPATH"

# Then start Python/Jupyter
python
# or
jupyter notebook
```

## Uninstall

If you want to uninstall the editable installation:

```bash
pip uninstall hydra-rp
```

## Development Tips

### 1. Fast Iteration

With editable install (`pip install -e .`), you can:
- Edit code
- Restart Jupyter kernel
- See changes immediately

### 2. Keep Dependencies Updated

```bash
# Update dependencies
pip install -r requirements.txt --upgrade
```

### 3. Test Your Changes

```bash
# Run a quick test
python -c "from hydra_rp import legal_reward_fn; print('OK')"
```

### 4. Check What's Installed

```bash
# See where the package is installed from
pip show hydra-rp

# Should show:
# Location: /path/to/HydraRP_Neko
# Editable project location: /path/to/HydraRP_Neko
```

## IDE Setup

### VS Code

Add to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ]
}
```

### PyCharm

1. File -> Settings -> Project -> Project Structure
2. Add Content Root: `/path/to/HydraRP_Neko`
3. Mark `hydra_rp` as Sources Root

## Common Issues

### Import Error

**Problem**: `ModuleNotFoundError: No module named 'hydra_rp'`

**Solution**: Make sure you've installed in editable mode:
```bash
pip install -e .
```

### Changes Not Reflected

**Problem**: Code changes don't appear

**Solutions**:
1. Restart Python interpreter or Jupyter kernel
2. Use `importlib.reload()` for specific modules
3. Check you edited the right file (not a cached `.pyc` file)

### Dependency Issues

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

## Summary

**Recommended for fast development:**
```bash
# One-time setup
cd /path/to/HydraRP_Neko
pip install -e .

# Then work normally
# Edit files -> Restart kernel -> Changes active!
```

This is the standard Python development workflow and works great with Jupyter notebooks!
