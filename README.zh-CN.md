# 月球南极区域巡视器路径规划

本仓库公开月球巡视器环境、论文实验数据以及数值结果图的绘图代码，用于检查环境、核验实验记录和复现论文图件。训练算法与训练检查点将在后续版本中补充。

## 仓库内容

| 目录 | 内容 |
| --- | --- |
| `src/lunar_rover_env/` | 月球巡视器环境的统一入口 |
| `vendor/gym_minigrid/` | 环境状态、观测、奖励、热模型和电量模型 |
| `vendor/gym/` | 环境使用的Gym 0.18.0运行依赖 |
| `data/environments/` | 坡度、太阳方向、光照和月表温度数据 |
| `results/raw/` | 各方法的轨迹记录和训练奖励 |
| `results/sensitivity/` | PPO超参数敏感性实验记录 |
| `results/repeated_runs/` | 三个场景各5次PPO+GRU奖励记录 |
| `results/tables/` | 与论文表格对应的汇总数据 |
| `plotting/` | 直接读取保存数据的绘图代码 |
| `manifests/` | 文件大小、SHA256和Release附件名称 |

## 论文图与数据的对应关系

| 论文图片 | 绘图代码读取的数据 |
| --- | --- |
| 图7 | `results/sensitivity/` |
| 图8 | `results/repeated_runs/site01`、`site06`和`site07` |
| 图9 | `results/raw/*/site01/`中各方法的数据，以及Site01坡度和光照数据 |
| 图10 | `results/raw/*/site06/`中各方法的数据，以及Site06坡度和光照数据 |
| 图11 | `results/raw/*/site07/`中各方法的数据，以及Site07坡度和光照数据 |
| 图12 | 图9对应的Site01轨迹，以及Site01光照和月表温度快照 |
| 图14 | `results/raw/*/site01_t660/`中的PPO、PPO+GRU和A*-shadow数据；环境数据采用Site01并偏移660个时间步 |
| 图15 | 图9和图14中T=0与T=660的对应轨迹 |
| 图16 | 图9、图14和图15对应的PPO+GRU轨迹，以及Site01月表温度图 |

`plotting/paper_figures.py`只读取这些已保存的CSV，不会重新运行学习算法。更详细的对应关系和数据格式见[`docs/FIGURE_DATA_MAP.md`](docs/FIGURE_DATA_MAP.md)和[`docs/DATA_CONVENTIONS.md`](docs/DATA_CONVENTIONS.md)。

## 环境地图数据附件

每个场景包含四类数据：

| 文件 | 用途 | 发布方式 |
| --- | --- | --- |
| `slope.csv` | 200 × 200坡度图，用于可通行性判断、坡度观测、奖励计算和轨迹图背景 | GitHub Release附件 |
| `Vsun.csv` | 1440个太阳方向采样，用于巡视器热功率模型 | GitHub Release附件 |
| `lunarSolarLight.csv` | 200 × 200地图随时间变化的光照，用于环境状态和论文光照曲线／快照 | GitHub Release附件 |
| `lunarSurfaceTemp.csv` | 200 × 200地图随时间变化的月表温度，用于环境热计算以及图12、图16 | GitHub Release附件 |

下载Release附件，并根据`manifests/release_assets.json`中的SHA-256校验值验证文件：

```bash
python scripts/download_data.py --repository OWNER/REPOSITORY
python scripts/check_data.py --sha256
```

下载脚本会自动解压Release附件，将CSV放入环境和绘图代码所需的目录，并校验原始文件。如只需一个场景，可增加`--site site01`、`--site site06`或`--site site07`进行下载和验证。数据形状、单位和时间索引见[`data/README.md`](data/README.md)。

## 使用方法

建议使用Python 3.10。

```bash
python -m venv .venv
# 根据操作系统激活虚拟环境。
python -m pip install -r requirements.txt

# 核对实验记录与论文表格。
python scripts/verify_paper_results.py

# 图7和图8只需要Git仓库中的小文件。
python plotting/paper_figures.py --figures 7 8

# 下载环境地图附件后：
python scripts/smoke_env.py --site site01 --steps 2
python plotting/paper_figures.py --figures 9 10 11 12 14 15 16
```

绘图结果保存在`outputs/paper_figures/`，包括高分辨率PNG、SVG、PDF、预览图和独立子图。论文中的最终组合图可在排版软件中进一步调整子图间距和标注。

第三方代码来源与许可证见[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
