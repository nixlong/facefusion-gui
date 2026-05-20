#!/usr/bin/env python3

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'

_ACCELERATORS_LOADED = False


def _init_accelerators():
    global _ACCELERATORS_LOADED
    if _ACCELERATORS_LOADED:
        return
    _ACCELERATORS_LOADED = True
    print(f"[Accelerator] _init_accelerators() pid={os.getpid()} "
          f"frozen={getattr(sys, 'frozen', False)} "
          f"MEIPASS={getattr(sys, '_MEIPASS', 'N/A')}", file=sys.stderr)
    try:
        from app.core.accelerator_manager import init_accelerators
        loaded = init_accelerators()
        print(f"[Accelerator] init done, loaded={len(loaded)} items", file=sys.stderr)
        for a in loaded:
            print(f"[Accelerator]   - {a['name']} providers={a['providers']}", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"[Accelerator] init failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


_init_accelerators()

from facefusion import conda, core

class FaceFusionApp:
	
	@staticmethod
	def facefusion_main():
		"""
		函数式启动 FaceFusion 核心
		自动构造命令行参数 run，兼容底层所有逻辑
		"""
		# ==============================================
		# 关键：手动设置命令行参数
		# 模拟执行：python main.py run
		# 让后续 conda.setup() / core.cli() 正常读取 run 参数
		# ==============================================
		os.environ['OMP_NUM_THREADS'] = '1'
		os.environ['FACEOFF_MASTER']  = '1'

		sys.argv = [sys.argv[0], "run"]

		conda.setup()
		core.cli()
		pass