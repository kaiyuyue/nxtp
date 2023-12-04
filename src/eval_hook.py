import sys
import importlib

if __name__ == "__main__":
    """
    Usage:
        ./scripts/run eval_hook.py ${eval_python_file} ${eval_config_file}
    """
    argv = sys.argv[1:]  # rm the script self name
    if len(argv) == 2:
        # for lang classifier
        run_path = str(argv[0]).replace("/", ".").replace(".py", "")
        cfg_path = str(argv[1]).replace("/", ".").replace(".py", "")

        run = importlib.import_module(run_path)
        cfg = importlib.import_module(cfg_path)

        run.main(cfg)
