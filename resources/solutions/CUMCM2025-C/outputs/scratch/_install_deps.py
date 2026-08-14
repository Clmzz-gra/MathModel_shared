import subprocess, sys, os
log_path = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/scratch/install_log.txt"
with open(log_path, 'w') as f:
    f.write("Starting install...\n")
    f.flush()
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "umap-learn", "hdbscan"],
        capture_output=True, text=True, timeout=300
    )
    f.write(f"Return code: {r.returncode}\n")
    f.write(f"STDOUT:\n{r.stdout}\n")
    f.write(f"STDERR:\n{r.stderr}\n")
    f.flush()

print(f"Done. Log at: {log_path}")
try:
    import umap; print("umap:", umap.__version__)
except Exception as e:
    print("umap FAILED:", e)
try:
    import hdbscan; print("hdbscan OK")
except Exception as e:
    print("hdbscan FAILED:", e)
