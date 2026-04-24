#!/usr/bin/env python3

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'

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

		sys.argv = [sys.argv[0], "run"]

		conda.setup()
		core.cli()
		pass