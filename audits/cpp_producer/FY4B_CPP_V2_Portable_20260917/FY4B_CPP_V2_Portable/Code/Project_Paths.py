"""
Description: 项目公共路径与资源定位工具。
Author: Yangzx
LastEditTime: 2026-07-17
Version:
    V1.1_122+local.1
Change:
    1. 提供项目根目录、Model、Auxiliary_Data 和 Kuang_Time 等公共路径定位功能。
    2. local.1: 结果路径支持环境变量覆盖，默认值保持服务器 V1.1_122 兼容。
    3. local.1: 补充路径优先级、只定位不创建目录的行为说明；逻辑不变。
"""

import os
from pathlib import Path

# 项目根目录由本文件位置推导，不依赖启动时的当前工作目录。
Project_Root = Path(__file__).resolve().parent.parent
Code_Dir = Project_Root / "Code"
# 优先级：专项环境变量 > GLOBAL_CPP_RESOURCE_ROOT 下的默认子目录 > 本地项目目录。
Resource_Root = Path(os.getenv('GLOBAL_CPP_RESOURCE_ROOT', str(Project_Root)))
Auxiliary_Data_Dir = Path(os.getenv('GLOBAL_CPP_AUXILIARY_ROOT', str(Resource_Root / "Auxiliary_Data")))
Model_Dir = Path(os.getenv('GLOBAL_CPP_MODEL_ROOT', str(Resource_Root / "Model")))
Kuang_Time_Dir = Path(os.getenv('GLOBAL_CPP_KUANG_TIME_ROOT', str(Resource_Root / "Kuang_Time")))
Test_Dir = Path(os.getenv('GLOBAL_CPP_TEST_ROOT', str(Project_Root / 'Test')))
Result_Dir = Path(os.getenv('GLOBAL_CPP_RESULT_ROOT', str(Test_Dir / 'Result')))
Results_Dir = Path(os.getenv('GLOBAL_CPP_GLOBAL_RESULTS', str(Test_Dir / 'Results')))
Check_Dir = Path(os.getenv('GLOBAL_CPP_CHECK_DIR', str(Test_Dir / 'Check')))

def Get_Auxiliary_Data_Path(File_Name):
    """返回 Auxiliary_Data 目录下的文件路径。"""
    return Auxiliary_Data_Dir / File_Name


def Get_Model_Path(Category, File_Name):
    """返回 Model 目录下指定分类模型文件的路径。"""
    return Model_Dir / Category / File_Name


def Get_Kuang_Time_Path(File_Name):
    """返回 Kuang_Time 目录下的文件路径。"""
    return Kuang_Time_Dir / File_Name


def Get_Satellite_Result_Root(Category, Name):
    """返回单颗卫星结果根目录。"""
    return Result_Dir / Category / Name / "Result"


def Get_Global_Results_Dir():
    """返回全球融合结果输出目录。"""
    return Results_Dir


def Get_Check_Dir():
    """返回完整性检查报告目录；检查模块只读结果目录。"""
    return Check_Dir
