try:
    import gurobipy as gp
    # 打印Gurobi版本号（调用version方法）
    print("Gurobi版本号：", gp.gurobi.version())
    # 尝试创建空模型，验证核心功能可用
    model = gp.Model("test_model")
    print("✅ Gurobi安装成功，核心功能可用！")
except ImportError:
    print("❌ 未找到gurobipy库，安装失败！")
except Exception as e:
    print(f"⚠️ Gurobi库存在，但运行出错：{e}")