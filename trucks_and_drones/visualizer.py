import sys
import os
# ─── 在导入 pygame 之前设置，保证窗口水平+垂直居中 ─────────────────────────────────
os.environ['SDL_VIDEO_CENTERED'] = '1'
os.environ['SDL_VIDEO_WINDOW_POS']    = 'center'

import numpy as np
import pygame
from pygame import Surface
import time



class BaseVisualizer:
    def __init__(self, name, visual_params, temp_db, enabled=True):  # 添加enabled参数
        self.enabled = enabled
        self.name = name
        self.temp_db = temp_db
        self.grid = temp_db.grid
        
        # 如果未启用可视化，提前返回
        if not self.enabled:
            self.no_open_window = True
            self.paused = False
            print("📊 Visualization disabled for this session")
            return

        # Initialize pygame
        pygame.init()

        self.no_open_window = True
        self.paused = False
        
        # Define parameter:
        self.name = name
        self.temp_db = temp_db
        self.grid = temp_db.grid
        [setattr(self, k, v) for k, v in visual_params.items()]

        # ===================== 新增：路径追踪相关 =====================
        # 存储每个episode的完整路径
        self.episode_paths = {}  # {episode: {'trucks': [], 'drones': []}}
        self.current_episode = 0
        self.current_step = 0
        
        # 当前episode的路径缓存
        self.truck_paths = []  # 每个truck的路径 [[step0_pos, step1_pos, ...], ...]
        self.drone_paths = []  # 每个drone的路径 [[step0_pos, step1_pos, ...], ...]

        # 正确初始化卡车路径列表
        for i in range(self.temp_db.num_trucks):
            self.truck_paths.append([])
            
        # 正确初始化无人机路径列表
        for i in range(self.temp_db.num_drones):
            self.drone_paths.append([])
        # ================================================================

        # ===================== [修改1] 放大网格尺寸 =====================
        ow, oh = visual_params['grid_surface_dim']
        # 将缩放系数从 0.6 改为 0.9，让地图更大
        scale_factor = 0.6 
        visual_params['grid_surface_dim'] = [int(ow * scale_factor), int(oh * scale_factor)]
        self.grid_surface_dim_original = [int(ow * scale_factor), int(oh * scale_factor)]

        # Define some colors
        self.color_dict = {
            'no_items_white': (240, 240, 240),
            'white': (255, 255, 255),
            'light-grey': (195, 195, 195),
            'grey': (128, 128, 128),
            'black': (0, 0, 0),
            'half_transp': (255, 255, 255, 125),
            'full_transp': (255, 255, 255, 0),
            'red': (165, 36, 36),
            'green': (67, 149, 64),
            'blue': (81, 73, 186),
            'purple': (151, 69, 176),
            'light-blue': (65, 163, 212),
            'orange': (239, 179, 110),
            'yellow': (239, 203, 24),
            'dark_red': (139, 0, 0),
            'dark_green': (0, 100, 0),
            'pink': (255, 192, 203),
            'inactive_dynamic': (100, 100, 100),      # 灰色 - 未激活的动态节点
            'active_dynamic': (0, 200, 100),          # 青绿色 - 激活的动态节点
            'static_customer': (67, 149, 64),         # 原绿色 - 静态客户节点
            'served_static': (128, 128, 128),         # 深灰色 - 已服务的静态节点
            'served_dynamic': (150, 150, 150),        # 浅灰色 - 已服务的动态节点
            'dynamic_border': (255, 165, 0),          # 橙色边框 - 动态节点标识
            'time_critical': (255, 100, 100),         # 红色 - 接近deadline的节点
        }
        # ========== 新增：动态生成载具颜色 ==========
        num_vehicles = len(self.temp_db.status_dict.get('v_coord', []))
        self.truck_colors = self._generate_vehicle_colors(num_vehicles, 'truck')
        self.drone_colors = self._generate_vehicle_colors(num_vehicles, 'drone')
        print(f"🎨 Generated colors for {num_vehicles} vehicles:")
        print(f"   Truck colors: {self.truck_colors}")
        print(f"   Drone colors: {self.drone_colors}")
        # ============================================

        # 缩小marker_size节点图标
        # self.marker_size = max(4, int(self.marker_size * 0.7))  # 缩小但不低于4
        self.marker_size = 12

        # Grid calculations
        self.x_mulipl = int(round(self.grid_surface_dim_original[0] / (self.grid[0])))
        self.y_mulipl = int(round(self.grid_surface_dim_original[1] / (self.grid[1])))

        self.axis_size = 20  # 稍微增加坐标轴空间
        # [修改] 计算实际网格显示区域大小 (包含padding)
        # 这里的原理是：给宽度额外增加 80 像素的隐形空间，让文字有地方画
        extra_right_padding = 50

        # 计算实际网格显示区域大小 (包含padding)
        self.grid_surface_dim = [
            self.grid_surface_dim_original[0] + (self.marker_size * 4) + extra_right_padding, # 👈 关键修改：加上这个间距
            self.grid_surface_dim_original[1] + (self.marker_size * 6)
        ]

        # ===================== 新增：视图控制变量 =====================
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_dragging = False
        self.last_mouse_pos = None
        self.grid_padding = max(10, int(getattr(self, 'grid_padding', 20) * 0.8))
        # ============================================================
 
        # ===================== [修改2] 布局尺寸计算 =====================
        
        # 1. 定义 Log 面板的宽度 (显示在地图右侧)
        self.log_panel_width = 320  # 给文字留足空间
        
        # 2. 计算上半部分（地图 + 坐标轴 + Log面板）的总宽度
        # 原来是: self.top_section_width = self.grid_surface_dim[0] + self.axis_size
        # 现在加上 Log 面板
        self.map_section_width = self.grid_surface_dim[0] + self.axis_size
        self.top_section_width = self.map_section_width + self.log_panel_width

        extra_log_height = 50  # 您可以根据需要调整这个数值，比如 100 或 150
        self.top_section_height = self.grid_surface_dim[1] + self.axis_size + extra_log_height
        # ================= [修改结束] =================

        # self.top_section_height = self.grid_surface_dim[1] + self.axis_size

        # 3. 定义底部区域的高度
        self.bottom_panel_height = 220  # 足够容纳图例和状态信息

        # 4. 分配底部宽度
        # 状态面板 (Status) 放在左下角
        status_width = 165 
        
        # 图例面板 (Info/Legends) 放在右下角，填满剩余宽度，保证对齐
        # 如果计算出的宽度太小，至少保留600像素以免图例挤在一起
        info_width = max(450, self.top_section_width - status_width) 
        self.bottom_section_width = status_width + info_width

        # 5. [新增] 计算居中对齐的偏移量
        # 找出上下两部分谁更宽
        self.max_content_width = max(self.top_section_width, self.bottom_section_width)
        
        # 计算地图需要的水平偏移量 (使其相对于底部居中)
        self.map_offset_x = (self.max_content_width - self.top_section_width) // 2
        
        # 计算底部面板需要的水平偏移量 (使其相对于地图居中 - 通常是0，因为底部往往更宽)
        self.bottom_offset_x = (self.max_content_width - self.bottom_section_width) // 2

        # 5. 计算窗口总宽度
        total_content_width = self.max_content_width

        # Initialize Surfaces
        # A. 地图层
        self.grid_surface = Surface(self.grid_surface_dim, pygame.SRCALPHA)
        self.travel_surface = Surface(self.grid_surface_dim, pygame.SRCALPHA)
        
        # B. 坐标轴背景层 (尺寸与地图层配合)
        self.grid_info_surface = Surface([
            self.grid_surface_dim[0] + self.axis_size,
            self.grid_surface_dim[1] + self.axis_size
        ], pygame.SRCALPHA)

        # [新增] C. Mission Log Surface (放在地图右侧)
        self.log_surface = Surface([self.log_panel_width, self.top_section_height], pygame.SRCALPHA)

        # C. 底部面板
        # Status Surface (左下)
        self.status_surface = Surface([status_width, self.bottom_panel_height], pygame.SRCALPHA)
        # Info Surface (右下 - 图例)
        self.info_surface = Surface([info_width, self.bottom_panel_height], pygame.SRCALPHA)

        # Window Size Calculation
        margin_x = 30
        margin_y = 30
        
        # 1. 获取当前屏幕的物理分辨率
        display_info = pygame.display.Info()
        screen_w = display_info.current_w
        screen_h = display_info.current_h
        
        # 2. 设定最大窗口尺寸 (屏幕尺寸减去一些边距，防止被任务栏遮挡)
        max_window_w = screen_w - 50
        max_window_h = screen_h - 70  # 比如 1080 - 100 = 980，保证能完整显示
        # 3. 计算理想的窗口尺寸
        ideal_width = total_content_width + 2 * margin_x
        ideal_height = self.top_section_height + self.bottom_panel_height + 2 * margin_y
        
        # 4. 应用限制
        self.window_width = min(max_window_w, ideal_width)
        self.window_height = min(max_window_h, ideal_height)
        
        # 5. [关键] 如果地图太高导致屏幕装不下，强制调整起始位置，防止底部被裁切
        # 如果内容高度 > 窗口高度，说明屏幕装不下全图了
        if ideal_height > self.window_height:
            print("⚠ Warning: Map is too tall for this screen! Resizing content...")
            # 这种情况下，我们优先保证能看到部分地图和完整的底部面板
            # 但 Pygame 不支持滚动，所以最好的办法是建议你减小 scale_factor
            pass

        # 计算内容起始位置
        # 1. 水平方向：保持居中 (Window宽 - 内容宽) / 2
        self.content_start_x = (self.window_width - total_content_width) // 2
        
        # 2. [修改] 垂直方向：将内容整体上移
        # 原代码: self.content_start_y = (self.window_height - (...)) // 2  <-- 这是垂直居中
        # 新代码: 固定顶部距离为 5 像素，让地图尽可能靠上
        self.content_start_y =1
        
        # 确保不小于边界
        self.content_start_x = max(10, self.content_start_x)
        # self.content_start_y = max(10, self.content_start_y) # 这一行不需要了，因为上面已经硬性指定为 5

        # Fonts
        self.big_font = pygame.font.SysFont('Arial', 19, bold=True)
        self.medium_font = pygame.font.SysFont('Arial', 16, bold=False)
        self.small_font = pygame.font.SysFont('Arial', 13, bold=False)
        self.tiny_font = pygame.font.SysFont('Arial', 11, bold=False)

        # Create window
        self.create_window()
        
        # Initial Draw
        self.reset_surfaces()
        self.draw_nodes()
        self.draw_vehicles()
        # self.draw_legends()
        try:
            self.draw_status_info(0, 0)
        except:
            pass

        # ===================== [修改3] 初始 Blit 逻辑 =====================
        # 使用新的布局坐标进行第一次绘制
        grid_x = self.content_start_x
        grid_y = self.content_start_y
        
        # 1. 绘制地图部分 (Top)
        # 坐标轴层
        self.screen.blit(self.grid_info_surface, (grid_x, grid_y))
        # 实体层
        self.screen.blit(self.grid_surface, (grid_x, grid_y))
        # 路径层
        self.screen.blit(self.travel_surface, (grid_x, grid_y))
        
        # 2. 绘制底部部分 (Bottom)
        bottom_y = grid_y + self.top_section_height
        
        # 左侧: 状态面板
        self.screen.blit(self.status_surface, (grid_x, bottom_y))
        
        # 右侧: 图例面板 (紧贴状态面板右侧)
        status_w = self.status_surface.get_width()
        self.screen.blit(self.info_surface, (grid_x + status_w, bottom_y))
        
        pygame.display.flip()
        

    # 2. 新增方法：动态生成载具颜色
    def _generate_vehicle_colors(self, num_vehicles, vehicle_type):
        """为不同数量的载具生成区分度高的颜色"""
        colors = []
        
        if vehicle_type == 'truck':
            # 卡车使用暖色调（红、橙、紫红、深红等）
            base_colors = [
                (200, 50, 50),   # 明亮红
                (255, 100, 0),   # 橙红
                (180, 0, 100),   # 紫红
                (139, 0, 0),     # 深红
                (255, 140, 0),   # 深橙
                (160, 32, 120),  # 紫色
                (205, 92, 92),   # 印第安红
                (220, 20, 60),   # 猩红
            ]
        else:  # drone
            # 无人机使用冷色调（蓝、青、蓝绿、深蓝等）
            base_colors = [
                (0, 100, 200),   # 明亮蓝
                (0, 191, 255),   # 深天蓝
                (70, 130, 180),  # 钢蓝
                (0, 0, 139),     # 深蓝
                (0, 139, 139),   # 深青
                (32, 178, 170),  # 浅海蓝绿
                (106, 90, 205),  # 石板蓝
                (65, 105, 225),  # 皇家蓝
            ]
        # 根据载具数量调整颜色
        for i in range(num_vehicles):
            if i < len(base_colors):
                colors.append(base_colors[i])
            else:
                # 超出预设颜色时，基于HSV生成新颜色
                if vehicle_type == 'truck':
                    # 卡车：红色系 (H: 0-60度)
                    h = (i * 37) % 60  # 使用质数37避免重复
                    s = 0.8 + (i % 3) * 0.1  # 饱和度 0.8-1.0
                    v = 0.7 + (i % 4) * 0.075  # 明度 0.7-0.925
                else:
                    # 无人机：蓝色系 (H: 180-280度)
                    h = 180 + (i * 41) % 100  # 蓝色到紫色范围
                    s = 0.7 + (i % 4) * 0.075
                    v = 0.6 + (i % 5) * 0.08
                
                # HSV转RGB
                import colorsys
                r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
                colors.append((int(r*255), int(g*255), int(b*255)))
        
        return colors
    
    def create_window(self):
        """创建 pygame 窗口"""
        if self.no_open_window:
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption(f"VRPD Simulation - {self.name}")
            self.no_open_window = False
            print("🖥️  Visualization window created")

        
    def reset_surfaces(self):
        """重置所有表面"""
        self.grid_surface.fill(self.color_dict['full_transp'])
        self.grid_info_surface.fill(self.color_dict['full_transp'])
        self.travel_surface.fill(self.color_dict['full_transp'])
        self.info_surface.fill(self.color_dict['full_transp'])
        self.status_surface.fill(self.color_dict['full_transp'])
        self.screen.fill(self.color_dict['white'])

        # Draw grid background
        for i in range(self.grid[0] + 1):
            start_x = self.axis_size + self.marker_size * 2 + i * self.x_mulipl
            pygame.draw.line(self.grid_info_surface, self.color_dict['light-grey'],
                             (start_x, self.axis_size + self.marker_size * 2),
                             (start_x, self.axis_size + self.marker_size * 2 + self.grid[1] * self.y_mulipl))

        for i in range(self.grid[1] + 1):
            start_y = self.axis_size + self.marker_size * 2 + i * self.y_mulipl
            pygame.draw.line(self.grid_info_surface, self.color_dict['light-grey'],
                             (self.axis_size + self.marker_size * 2, start_y),
                             (self.axis_size + self.marker_size * 2 + self.grid[0] * self.x_mulipl, start_y))


    def draw_nodes(self):
        """绘制节点 - 修改：将Depot绘制为房子形状"""
        try:
            coords = self.temp_db.get_val('n_coord')
            delta = self.temp_db.get_val('delta')
            deadlines = self.temp_db.get_val('deadline')
            current_time = getattr(self.temp_db, 'current_time', 0.0)
            
            # 获取道路破损信息
            road_damaged = self.temp_db.get_val('road_damaged')
            damaged_nodes = set(i for i, damaged in enumerate(road_damaged) if damaged == 1)
            
            static_customer_start = self.temp_db.num_depots
            static_customer_end = static_customer_start + getattr(self.temp_db, 'num_static_customers', self.temp_db.num_customers)
            
            for i, (x, y) in enumerate(coords):
                # 跳过未激活的动态节点
                if delta[i] == -1:
                    continue
                
                # 判断是否实际被服务
                is_actually_served = (i in self.temp_db.visited_nodes)

                # 转换坐标
                screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)
                
                # --- 分支 1: 绘制 Depot (房子形状) ---
                if i in self.temp_db.depot_indices:
                    color = self.color_dict['orange']
                    node_type = f"Depot"
                    
                    # 1.【修改】增大尺寸
                    # 原来是 +4，现在改为 +8，整体大一圈
                    ms = self.marker_size + 8 
                    
                    # 2.【修改】定义垂直的房子形状
                    # 关键点：右屋檐和右底角的 X 坐标必须一致，墙壁才会垂直
                    # 这里统一使用 ms * 0.9 作为半宽度，即屋檐和墙脚对齐
                    
                    roof_top_y = ms         # 屋顶尖端向上偏移量
                    eave_y = ms * 0.3       # 屋檐处向上偏移量 (转折点)
                    floor_y = ms * 0.8      # 地板向下偏移量
                    half_width = ms * 0.9   # 房子的半宽

                    house_points = [
                        (screen_x, screen_y - roof_top_y),                  # 1. Top (屋顶尖端)
                        (screen_x + half_width, screen_y - eave_y),         # 2. Right Roof (右屋檐)
                        (screen_x + half_width, screen_y + floor_y),        # 3. Bottom Right (右墙脚) -> X与上面一致
                        (screen_x - half_width, screen_y + floor_y),        # 4. Bottom Left (左墙脚) -> X与上面一致
                        (screen_x - half_width, screen_y - eave_y)          # 5. Left Roof (左屋檐)
                    ]
                    
                    # 绘制实心房子
                    pygame.draw.polygon(self.grid_surface, color, house_points)
                    # 绘制房子边框 (黑色)
                    pygame.draw.polygon(self.grid_surface, self.color_dict['black'], house_points, 2)
                    
                    # 标签颜色
                    label_color = self.color_dict['white']

                # --- 分支 2: 绘制 Customer (圆形) ---
                else: 
                    # 1. 确定颜色和类型
                    if i < static_customer_end:  # 静态客户
                        if is_actually_served:
                            color = self.color_dict['served_static']
                            node_type = f"C{i}"
                        else:
                            time_remaining = deadlines[i] - current_time
                            if time_remaining < 30:
                                color = self.color_dict['time_critical']
                            else:
                                color = self.color_dict['static_customer']
                            node_type = f"C{i}"
                        marker_size = self.marker_size
                        border_color = self.color_dict['black']
                        border_width = 2
                        
                    else:  # 动态客户
                        if is_actually_served:
                            color = self.color_dict['served_dynamic']
                            node_type = f"D{i}"
                            marker_size = self.marker_size
                            border_color = self.color_dict['dynamic_border']
                            border_width = 3
                        else:
                            time_remaining = deadlines[i] - current_time
                            if time_remaining < 30:
                                color = self.color_dict['time_critical']
                            else:
                                color = self.color_dict['active_dynamic']
                            node_type = f"D{i}"
                            marker_size = self.marker_size + 1
                            border_color = self.color_dict['dynamic_border']
                            border_width = 3

                    # 道路破损样式覆盖
                    if i in damaged_nodes: 
                        color = self.color_dict['red']      # 红底
                        border_color = self.color_dict['red'] 
                        border_width = 2
                    
                    # 2. 绘制圆形实体
                    pygame.draw.circle(self.grid_surface, color, (screen_x, screen_y), marker_size)
                    pygame.draw.circle(self.grid_surface, border_color, (screen_x, screen_y), marker_size, border_width)
                    
                    # 标签颜色
                    label_color = self.color_dict['white'] if not is_actually_served else self.color_dict['black']

                # --- 通用: 绘制节点内部标签 (D0, C1...) ---
                label = self.tiny_font.render(node_type, True, label_color)
                # 调整：如果是房子，文字稍微下移一点点，视觉更平衡
                offset_y = 3 if i in self.temp_db.depot_indices else 0
                label_rect = label.get_rect(center=(screen_x, screen_y + offset_y))
                self.grid_surface.blit(label, label_rect)
                
                # --- 通用: 绘制右侧文本 (道路破损提示) ---
                if i not in self.temp_db.depot_indices and i in damaged_nodes:
                    damaged_label = self.tiny_font.render("Road Damaged!", True, self.color_dict['red'])
                    self.grid_surface.blit(damaged_label, (screen_x - 20, screen_y + 10))
            
        except Exception as e:
            print(f"❌ Error drawing nodes: {e}")
            import traceback
            traceback.print_exc()

    def _record_vehicle_positions(self):
        """记录当前步骤的车辆位置用于路径追踪 - 修复位置获取逻辑"""
        try:
            v_coords = self.temp_db.status_dict.get('v_coord', [])
            coords = self.temp_db.get_val('n_coord')

            num_trucks = self.temp_db.num_trucks
            num_drones = self.temp_db.num_drones
            
            # 记录truck位置（前 num_trucks 个车辆）
            for k in range(num_trucks):
                if k < len(v_coords) and k < len(self.truck_paths):
                    pos_index = v_coords[k]
                    if pos_index < len(coords):
                        x, y = coords[pos_index]
                        self.truck_paths[k].append((x, y))
            
            # 记录drone位置 - 关键修复：使用正确的位置获取逻辑
            ED = self.temp_db.status_dict.get('ED', [])
            ND = self.temp_db.status_dict.get('ND', [])
            
            for k in range(num_drones):
                if k < len(self.drone_paths) and k < len(ED):
                    # 获取无人机当前实际位置
                    current_pos = self._get_accurate_drone_position(k)
                    
                    if current_pos < len(coords):
                        x, y = coords[current_pos]
                        self.drone_paths[k].append((x, y))
                        
        except Exception as e:
            print(f"⚠ Error recording vehicle positions: {e}")
            import traceback
            traceback.print_exc()

    def _get_accurate_drone_position(self, drone_idx):
        """获取无人机的准确当前位置"""
        try:
            ED = self.temp_db.status_dict.get('ED', [])
            attached_truck = self.temp_db.status_dict.get('attached_truck', [])
            
            if drone_idx >= len(ED):
                return 0
                
            # 根据无人机状态确定位置
            if ED[drone_idx] == 3:  # 在卡车上
                if (drone_idx < len(attached_truck) and 
                    attached_truck[drone_idx] >= 0 and 
                    attached_truck[drone_idx] < self.temp_db.num_trucks):
                    return self.temp_db.status_dict['v_coord'][attached_truck[drone_idx]]
                else:
                    return 0
                    
            elif ED[drone_idx] in [0, 1, 2]:  # 在途中、等待、或刚完成服务
                # 优先使用独立坐标
                if 'drone_coord' in self.temp_db.status_dict:
                    if drone_idx < len(self.temp_db.status_dict['drone_coord']):
                        return self.temp_db.status_dict['drone_coord'][drone_idx]
                
                # 备用方案：使用v_coord中的位置
                vehicle_idx = self.temp_db.num_trucks + drone_idx
                if vehicle_idx < len(self.temp_db.status_dict.get('v_coord', [])):
                    return self.temp_db.status_dict['v_coord'][vehicle_idx]
                
                return 0
            else:
                return 0
                
        except Exception as e:
            print(f"Warning: Error getting drone {drone_idx} position: {e}")
            return 0

    
    # 新增辅助方法：绘制虚线
    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=6):
        """绘制虚线辅助方法"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return

        # 计算单位向量
        ux = dx / distance
        uy = dy / distance
        
        # 循环绘制实线段，跳过空隙
        # 模式: 实线(dash_length) -> 空隙(dash_length) -> 实线...
        curr_dist = 0
        while curr_dist < distance:
            # 线段起点
            p1_x = x1 + ux * curr_dist
            p1_y = y1 + uy * curr_dist
            
            # 线段终点（不能超过总长度）
            end_dist = min(curr_dist + dash_length, distance)
            p2_x = x1 + ux * end_dist
            p2_y = y1 + uy * end_dist
            
            pygame.draw.line(surface, color, (p1_x, p1_y), (p2_x, p2_y), width)
            
            # 前进到下一段（跳过一个间隙，间隙长度等于dash_length）
            curr_dist += dash_length * 2

    # 修改：增加 is_dashed 参数
    def _draw_path_with_arrows(self, path, color, line_width=3, arrow_color=None, is_dashed=False):
        """绘制带箭头的路径 - 支持虚线"""
        if len(path) < 2:
            return
            
        if arrow_color is None:
            arrow_color = color
            
        try:
            for i in range(len(path) - 1):
                start_pos = path[i]
                end_pos = path[i + 1]
                
                # 检查坐标格式
                if not (isinstance(start_pos, (tuple, list)) and len(start_pos) == 2):
                    continue
                if not (isinstance(end_pos, (tuple, list)) and len(end_pos) == 2):
                    continue
                    
                # 转换为屏幕坐标
                start_x, start_y = start_pos
                end_x, end_y = end_pos
                
                start_screen_x = int(self.axis_size + self.marker_size * 2 + start_x * self.x_mulipl)
                start_screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - start_y) * self.y_mulipl)
                end_screen_x = int(self.axis_size + self.marker_size * 2 + end_x * self.x_mulipl)
                end_screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - end_y) * self.y_mulipl)
                
                # 绘制路径线段
                if start_screen_x != end_screen_x or start_screen_y != end_screen_y:
                    if is_dashed:
                        # 如果是虚线模式，调用虚线绘制方法
                        self._draw_dashed_line(self.travel_surface, color,
                                             (start_screen_x, start_screen_y),
                                             (end_screen_x, end_screen_y), 
                                             line_width, dash_length=6)
                    else:
                        # 实线模式
                        pygame.draw.line(self.travel_surface, color,
                                    (start_screen_x, start_screen_y), 
                                    (end_screen_x, end_screen_y), line_width)
                    
                    # 绘制箭头（在线段中点）
                    self._draw_arrow(start_screen_x, start_screen_y, 
                                end_screen_x, end_screen_y, arrow_color)
                    
        except Exception as e:
            print(f"❌ Error drawing path: {e}")

    # 修改：为无人机路径设置 is_dashed=True
    def draw_paths(self):
        """绘制所有车辆的完整路径 - 使用动态颜色系统"""
        # 绘制truck路径（保持实线）
        for k, path in enumerate(self.truck_paths):
            if len(path) >= 2:
                truck_color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
                self._draw_path_with_arrows(path, truck_color, line_width=4, is_dashed=False)
        
        # 绘制drone路径（设置为虚线）
        for k, path in enumerate(self.drone_paths):
            if len(path) >= 2:
                drone_color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
                self._draw_path_with_arrows(path, drone_color, line_width=2, is_dashed=True)

    

    def draw_vehicles(self):
        """绘制车辆位置和状态"""
        try:
            # 卡车/无人机分开
            v_coords = self.temp_db.status_dict.get('v_coord')
            ET = self.temp_db.status_dict.get('ET')
            ED = self.temp_db.status_dict.get('ED')
            TW = self.temp_db.status_dict.get('TW')
            print("▶ TW in viz:", TW)
            DW = self.temp_db.status_dict.get('DW', [])  # 无人机负载
            LT = self.temp_db.status_dict.get('LT')
            LD = self.temp_db.status_dict.get('LD')
            NT = self.temp_db.status_dict.get('NT')
            ND = self.temp_db.status_dict.get('ND')

            # 【关键修改】这里漏掉了 attached_truck 的定义，必须补上！
            attached_truck = self.temp_db.status_dict.get('attached_truck')
            # 修正：使用正确的坐标键名
            coords = self.temp_db.get_val('n_coord')

            print(f"🚛 Drawing {len(v_coords)} vehicles:")
            print(f"   Vehicle positions (node indices): {v_coords}")
            print(f"   ET (truck status): {ET}")
            print(f"   ED (drone status): {ED}")

            # 分别处理卡车和无人机
            num_trucks = self.temp_db.num_trucks
            num_drones = self.temp_db.num_drones
            
            # 绘制卡车 (前 num_trucks 个车辆)
            for k in range(num_trucks):
                if k >= len(v_coords):
                    continue
                    
                # Get vehicle position from node coordinates
                pos_index = int(v_coords[k])

                if pos_index < len(coords):
                    x, y = coords[pos_index]
                    # 修正坐标转换，与节点绘制保持一致
                    screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                    screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)

                    # 获取基础颜色
                    base_truck_color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
                    
                    # --- 状态判定逻辑 ---
                    status_label = ""
                    truck_color = base_truck_color
                    
                    if k < len(ET):
                        target_node = NT[k]
                        is_serving_mission = (target_node != 0) # 目标不是depot，说明是去服务

                        if ET[k] == 0:  # In-transit (移动中)
                            if is_serving_mission:
                                # 【核心修改】去往客户节点途中，视为 "Serving" 过程
                                # 颜色加一点绿色分量表示服务中
                                truck_color = base_truck_color 
                                status_label = "Moving" 
                            else:
                                # 回depot或者其他移动
                                truck_color = base_truck_color
                                status_label = "Backing"
                                
                        elif ET[k] == 1:  # Waiting (等待中)
                            # 【核心修改】明确的等待颜色 (加黄色/变亮)
                            truck_color = base_truck_color
                            status_label = "Waiting"
                            
                        elif ET[k] == 2:  # Serving (到达瞬间)
                            truck_color = base_truck_color
                            status_label = "Serving"
                            
                        elif ET[k] == 3: # Idle
                            truck_color = base_truck_color
                            status_label = "Idle" # 空闲时不显示标签，保持画面整洁

                    # Draw truck as rectangle
                    truck_rect = pygame.Rect(screen_x - 10, screen_y - 20, 20, 15)
                    pygame.draw.rect(self.grid_surface, truck_color, truck_rect)
                    pygame.draw.rect(self.grid_surface, self.color_dict['black'], truck_rect, 2)

                    # Draw vehicle ID
                    id_label = self.tiny_font.render(f"T{k}", True, self.color_dict['white'])
                    id_rect = id_label.get_rect(center=(screen_x, screen_y - 12))
                    self.grid_surface.blit(id_label, id_rect)

                    # 【新增】在卡车下方绘制状态标签 (Serving/Waiting)
                    if status_label:
                        # 颜色逻辑
                        if status_label == "Waiting":
                            label_color = self.color_dict['dark_red']
                        elif "Serving" in status_label:
                            label_color = self.color_dict['dark_green']
                        elif "Idle" in status_label:
                            label_color = self.color_dict['dark_green']
                        else: # Backing
                            label_color = self.color_dict['yellow']

                        st_text = self.tiny_font.render(status_label, True, label_color)
                        st_rect = st_text.get_rect(center=(screen_x+30, screen_y-12)) # 显示上方
                        
                        # 画背景框
                        bg_rect = st_rect.inflate(6, 4) # 稍微大一点
                        # 1. 白底
                        pygame.draw.rect(self.grid_surface, (255, 255, 255, 230), bg_rect)
                        # 2. 【修改点4】黑色边框 (width=1)
                        pygame.draw.rect(self.grid_surface, self.color_dict['black'], bg_rect, 1)
                        # 3. 文字
                        self.grid_surface.blit(st_text, st_rect)

            # ================= 2. 绘制无人机 (Drones) =================
            for k in range(num_drones):
                vehicle_idx = num_trucks + k
                if vehicle_idx >= len(v_coords): continue
                
                # 获取无人机位置索引
                is_on_truck = (k < len(ED) and ED[k] == 3)
                pos_index = 0 # 默认值
                
                # 如果在卡车上，强制使用卡车位置，避免坐标不同步
                if is_on_truck and attached_truck is not None and k < len(attached_truck):
                    t_idx = attached_truck[k]
                    if t_idx >= 0 and t_idx < num_trucks:
                        # 如果在卡车上，直接取该卡车的位置
                        pos_index = int(v_coords[t_idx])
                    else:
                        pos_index = int(v_coords[vehicle_idx])
                else:
                    # 独立飞行或服务中，使用自身坐标
                    # 优先检查 drone_coord (更准确的独立坐标)
                    if 'drone_coord' in self.temp_db.status_dict and k < len(self.temp_db.status_dict['drone_coord']):
                        pos_index = self.temp_db.status_dict['drone_coord'][k]
                    else:
                        pos_index = int(v_coords[vehicle_idx])

                # 坐标转换
                if pos_index < len(coords):
                    x, y = coords[int(pos_index)]
                else:
                    x, y = 0, 0

                screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)

                # 视觉偏移：如果在卡车上，画在卡车上方一点，避免重叠
                if is_on_truck:
                    screen_y -= 15 

                # 颜色定义：固定颜色
                drone_color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
                
                # 状态判定
                status_label = ""
                label_color = self.color_dict['black']

                if k < len(ED):
                    if ED[k] == 3:  # On Truck
                        status_label = "On Truck"
                        label_color = self.color_dict['purple']
                    elif ED[k] == 1:  # Waiting
                        status_label = "Waiting"
                        label_color = self.color_dict['dark_red']
                    elif ED[k] == 2:  # Served / Idle
                        status_label = "Served"
                        label_color = self.color_dict['dark_green']
                    elif ED[k] == 0:  # In-transit (飞行中)
                        # 【核心修改】通过目标节点判断是 Serving 还是 Collection
                        target_node = ND[k]

                        # 安全获取 t_idx
                        t_idx = -1
                        if attached_truck is not None and k < len(attached_truck):
                             t_idx = attached_truck[k]

                        is_collection = False
                        
                        if t_idx != -1 and t_idx < num_trucks:
                            # 搭载模式：比较无人机目标和卡车汇合点
                            truck_et = ET[t_idx]
                            truck_pos = int(v_coords[t_idx])
                            truck_target = NT[t_idx]
                            
                            # 卡车的汇合点逻辑：如果卡车在移动，汇合点是目标；如果卡车静止，汇合点是当前位置
                            rendezvous_point = truck_target if truck_et == 0 else truck_pos
                            
                            # 如果无人机飞向卡车的汇合点，那就是去汇合(Collection)
                            if target_node == rendezvous_point:
                                is_collection = True
                        else:
                            # 独立模式：如果飞回 Depot(0)，视为 Collection/Return
                            if target_node == 0:
                                is_collection = True
                        
                        if is_collection:
                            status_label = "Collection" 
                            label_color = self.color_dict['orange']
                        else:
                            status_label = "Flying"
                            label_color = self.color_dict['dark_green']

                # 绘制无人机矩形 (20x15) - 形状同卡车
                drone_rect = pygame.Rect(screen_x - 10, screen_y - 20, 20, 15)
                pygame.draw.rect(self.grid_surface, drone_color, drone_rect)
                pygame.draw.rect(self.grid_surface, self.color_dict['black'], drone_rect, 2) # 黑色边框

                # 绘制 ID (DX)
                id_label = self.tiny_font.render(f"D{k}", True, self.color_dict['white'])
                id_rect = id_label.get_rect(center=(screen_x, screen_y - 12))
                self.grid_surface.blit(id_label, id_rect)

                # 绘制状态标签 
                if status_label:
                    st_text = self.tiny_font.render(status_label, True, label_color)
                    
                    # 调整标签位置：如果在卡车上，标签显示在上方，否则显示在下方
                    if is_on_truck:
                        st_rect = st_text.get_rect(center=(screen_x+32, screen_y - 17)) # 上方
                    else:
                        st_rect = st_text.get_rect(center=(screen_x+30, screen_y - 12)) # 下方
                    
                    # 背景框 + 边框
                    bg_rect = st_rect.inflate(6, 4)
                    pygame.draw.rect(self.grid_surface, (255, 255, 255, 230), bg_rect)
                    pygame.draw.rect(self.grid_surface, self.color_dict['black'], bg_rect, 1)
                    self.grid_surface.blit(st_text, st_rect)
                    
        except Exception as e:
            print(f"❌ Error drawing vehicles: {e}")
            import traceback
            traceback.print_exc()

    
    def _draw_arrow(self, start_x, start_y, end_x, end_y, color, size=8):
        """在两点间绘制箭头"""
        # 计算方向向量
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return
            
        # 单位方向向量
        ux = dx / length
        uy = dy / length
        
        # 箭头位置（在线段的70%处）
        arrow_pos_x = int(start_x + dx * 0.7)
        arrow_pos_y = int(start_y + dy * 0.7)
        
        # 箭头的两个翅膀点
        wing_length = size
        wing_angle = 0.6  # 约35度
        
        # 旋转向量得到箭头翅膀
        wing1_x = arrow_pos_x - int(wing_length * (ux * np.cos(wing_angle) - uy * np.sin(wing_angle)))
        wing1_y = arrow_pos_y - int(wing_length * (ux * np.sin(wing_angle) + uy * np.cos(wing_angle)))
        
        wing2_x = arrow_pos_x - int(wing_length * (ux * np.cos(-wing_angle) - uy * np.sin(-wing_angle)))
        wing2_y = arrow_pos_y - int(wing_length * (ux * np.sin(-wing_angle) + uy * np.cos(-wing_angle)))
        
        # 绘制箭头
        pygame.draw.polygon(self.travel_surface, color, [
            (arrow_pos_x, arrow_pos_y),
            (wing1_x, wing1_y),
            (wing2_x, wing2_y)
        ])

    
    # def draw_legends(self):
    #     """[新增] 在底部面板绘制图例：路径图例(分列) + 节点图例"""
    #     try:
    #         # 清空表面 (设为透明或背景色)
    #         # self.info_surface.fill(self.color_dict['full_transp']) 
            
    #         left_margin = 20
    #         col_width = 120  # 列宽
            
    #         # --- 1. Path Legend (分为两列：卡车、无人机) ---
    #         title_y = 5
    #         path_title = self.medium_font.render("Path Legend:", True, self.color_dict['purple'])
    #         self.info_surface.blit(path_title, (left_margin, title_y))

    #         item_start_y = title_y + 25
    #         line_y_offset = 8
            
    #         # 第一列：卡车 (Trucks)
    #         curr_y = item_start_y
    #         for k, _ in enumerate(self.truck_paths):
    #             color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
                
    #             # 文字 T0, T1...
    #             text = self.small_font.render(f"Truck {k}", True, self.color_dict['black'])
    #             self.info_surface.blit(text, (left_margin, curr_y))
                
    #             # 实线图标
    #             line_start_x = left_margin + 50
    #             line_end_x = line_start_x + 40
    #             pygame.draw.line(self.info_surface, color, (line_start_x, curr_y + line_y_offset), (line_end_x, curr_y + line_y_offset), 3)
                
    #             # 箭头装饰
    #             pygame.draw.polygon(self.info_surface, color, [
    #                 (line_end_x, curr_y + line_y_offset), 
    #                 (line_end_x - 5, curr_y + line_y_offset - 3), 
    #                 (line_end_x - 5, curr_y + line_y_offset + 3)
    #             ])
    #             curr_y += 18

    #         # 第二列：无人机 (Drones) - 位于卡车右侧
    #         drone_col_x = left_margin + col_width
    #         curr_y = item_start_y # 重置Y坐标，与卡车对齐
            
    #         for k, _ in enumerate(self.drone_paths):
    #             color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
                
    #             # 文字 D0, D1...
    #             text = self.small_font.render(f"Drone {k}", True, self.color_dict['black'])
    #             self.info_surface.blit(text, (drone_col_x, curr_y))
                
    #             # 虚线图标
    #             line_start_x = drone_col_x + 50
    #             line_end_x = line_start_x + 40
    #             self._draw_dashed_line(self.info_surface, color, 
    #                                  (line_start_x, curr_y + line_y_offset), 
    #                                  (line_end_x, curr_y + line_y_offset), 
    #                                  width=2, dash_length=4)
                
    #             # 箭头装饰
    #             pygame.draw.polygon(self.info_surface, color, [
    #                 (line_end_x, curr_y + line_y_offset), 
    #                 (line_end_x - 5, curr_y + line_y_offset - 3), 
    #                 (line_end_x - 5, curr_y + line_y_offset + 3)
    #             ])
    #             curr_y += 18

    #         # --- 2. Node Legend (位于路径图例右侧) ---
    #         # 计算节点图例的起始X坐标 (根据前两列宽度)
    #         node_legend_x = drone_col_x + col_width + 10
            
    #         title_node = self.medium_font.render("Node Legend:", True, self.color_dict['dark_green'])
    #         self.info_surface.blit(title_node, (node_legend_x, title_y))
            
    #         curr_y = item_start_y
            
    #         legend_items = [
    #             (self.color_dict['orange'], self.color_dict['black'], 2, 0, "Depot"),
    #             (self.color_dict['static_customer'], self.color_dict['black'], 2, 0, "Customer"),
    #             (self.color_dict['served_static'], self.color_dict['black'], 2, 0, "Served"),
    #             (self.color_dict['active_dynamic'], self.color_dict['dynamic_border'], 3, 1, "Dynamic Request"),
    #             (self.color_dict['red'], self.color_dict['red'], 2, 2, "Road Damaged Customer")
    #         ]

    #         for fill_col, border_col, border_w, shape_type, desc in legend_items:
    #             icon_x = node_legend_x + 8
    #             icon_y = curr_y + 7
    #             rad = 6
                
    #             # 绘制图标 (复用之前的逻辑)
    #             if shape_type == 2: # Road Damaged
    #                 if fill_col: pygame.draw.circle(self.info_surface, fill_col, (icon_x, icon_y), rad)
    #                 if border_col: pygame.draw.circle(self.info_surface, border_col, (icon_x, icon_y), rad, border_w)
    #                 pygame.draw.line(self.info_surface, border_col, (icon_x - rad, icon_y - rad), (icon_x + rad, icon_y + rad), 2)
    #                 pygame.draw.line(self.info_surface, border_col, (icon_x - rad, icon_y + rad), (icon_x + rad, icon_y - rad), 2)
    #             else:
    #                 if fill_col: pygame.draw.circle(self.info_surface, fill_col, (icon_x, icon_y), rad)
    #                 if border_col: pygame.draw.circle(self.info_surface, border_col, (icon_x, icon_y), rad, border_w)
    #                 if shape_type == 1: 
    #                     pygame.draw.circle(self.info_surface, self.color_dict['white'], (icon_x, icon_y), 2)

    #             text = self.small_font.render(desc, True, self.color_dict['black'])
    #             self.info_surface.blit(text, (node_legend_x + 25, curr_y))
    #             curr_y += 18

    #     except Exception as e:
    #         print(f"Error drawing legends: {e}")

    
    # def draw_status_info(self, episode, step, last_actions=None, last_rewards=None):
    #     """绘制状态信息面板"""
    #     y_offset = 10
    #     line_height = 18
    #     left_margin = 10  

    #     # Title
    #     title = self.big_font.render("VRPD Status", True, self.color_dict['black'])
    #     self.status_surface.blit(title, (10, y_offset))
    #     y_offset += 30

    #     # Episode and step info
    #     episode_text = self.medium_font.render(f"Episode: {episode}", True, self.color_dict['black'])
    #     self.status_surface.blit(episode_text, (10, y_offset))
    #     y_offset += line_height

    #     step_text = self.medium_font.render(f"Step: {step}", True, self.color_dict['black'])
    #     self.status_surface.blit(step_text, (10, y_offset))
    #     y_offset += line_height

    #     # # ========== 新增：显示总训练步数 ==========
    #     # total_steps = getattr(self, 'total_training_steps', 0)
    #     # total_step_text = self.medium_font.render(f"Total Steps: {total_steps}", True, self.color_dict['dark_green'])
    #     # self.status_surface.blit(total_step_text, (10, y_offset))
    #     # y_offset += line_height + 5
    #     # # =====================================

    #     # Time info
    #     try:
    #         total_time = getattr(self.temp_db, 'total_time', 0.0)
    #         time_text = self.medium_font.render(f"Time: {total_time:.2f}", True, self.color_dict['black'])
    #         self.status_surface.blit(time_text, (10, y_offset))
    #         y_offset += line_height
    #     except:
    #         pass
        
    #     # 5. Node Statistics (简化显示逻辑)
    #     try:
    #         # nodes_title = self.medium_font.render("Node Status:", True, self.color_dict['dark_green'])
    #         # self.status_surface.blit(nodes_title, (left_margin, y_offset))
    #         # y_offset += line_height

    #         # 【修改点3】：计算 Visited/Total (均不包含 Depot)
    #         delta = self.temp_db.get_val('delta')
            
    #         # 获取 Depot 索引集合 (通常是 [0])
    #         if hasattr(self.temp_db, 'depot_indices'):
    #             depots = set(self.temp_db.depot_indices)
    #         else:
    #             depots = {0} # 默认 0 是 depot
            
    #         # 分母：总节点数 - Depot数
    #         total_customers = len(delta) - len(depots)
            
    #         # 分子：已访问集合中 剔除 Depot
    #         # visited_nodes 只有在车辆实际到达节点时才会添加 (见之前的 draw_nodes 逻辑)
    #         visited_customers = sum(1 for n in self.temp_db.visited_nodes if n not in depots)

    #         visited_text = self.medium_font.render(f"Compleded served: {visited_customers}/{total_customers}", True, self.color_dict['black'])
    #         self.status_surface.blit(visited_text, (left_margin, y_offset))
    #         y_offset += line_height + 5

    #     except Exception as e:
    #         pass

    #     except Exception as e:
    #         error_text = self.small_font.render(f"Vehicle Status Error: {str(e)[:30]}", True, self.color_dict['red'])
    #         self.status_surface.blit(error_text, (10, y_offset))
    #         y_offset += line_height

        # # Last rewards —— 兼容标量 / list / ndarray；避免直接用 if last_rewards:
        # if last_rewards is not None:
        #     try:
        #         r_arr = np.asarray(last_rewards, dtype=float).reshape(-1)
        #     except Exception:
        #         # 无法转成数组就当作标量
        #         try:
        #             r_arr = np.array([float(last_rewards)], dtype=float)
        #         except Exception:
        #             r_arr = np.array([], dtype=float)

        #     if r_arr.size > 0:
        #         y_offset += 5
        #         rewards_title = self.medium_font.render("Last Rewards:", True, self.color_dict['dark_red'])
        #         self.status_surface.blit(rewards_title, (10, y_offset))
        #         y_offset += line_height

        #         for i, reward in enumerate(r_arr):
        #             reward_text = self.small_font.render(f"  Agent {i}: {float(reward):.3f}", True, self.color_dict['black'])
        #             self.status_surface.blit(reward_text, (20, y_offset))
        #             y_offset += 16

        # # Control instructions
        # y_offset += 10
        # control_title = self.medium_font.render("Controls:", True, self.color_dict['purple'])
        # self.status_surface.blit(control_title, (10, y_offset))
        # y_offset += line_height

        # instructions = [
        #     "SPACE: Pause/Resume",
        #     "ENTER: Next Step (when paused)",
        #     "ESC: Quit",
        #     "Click: Node details"
        # ]

        # for instruction in instructions:
        #     instr_text = self.small_font.render(instruction, True, self.color_dict['black'])
        #     self.status_surface.blit(instr_text, (20, y_offset))
        #     y_offset += 16


    def draw_mission_log(self):
        """
        绘制右侧面板：包含上半部分的文字日志 和 下半部分的任务看板
        """
        # 1. 清空面板背景
        self.log_surface.fill(self.color_dict['white']) 
        
        # 绘制左侧分割线
        pygame.draw.line(self.log_surface, self.color_dict['light-grey'], (0, 0), (0, self.top_section_height), 2)

        # ==========================================
        # Part A: Real-time Text Log (上半部分)
        # ==========================================
        
        # 标题栏
        title_y = 5
        title = self.big_font.render("Mission Log (Real-time):", True, self.color_dict['black'])
        self.log_surface.blit(title, (10, title_y))

        # 显示 Step 和 Time
        current_step = getattr(self, 'current_step', 0)
        total_time = getattr(self.temp_db, 'total_time', 0.0)
        
        step_text = self.medium_font.render(f"Step: {current_step}", True, self.color_dict['dark_green'])
        time_text = self.medium_font.render(f"Time: {total_time:.2f} min", True, self.color_dict['dark_green'])
        
        self.log_surface.blit(step_text, (10, 30))
        self.log_surface.blit(time_text, (140, 30))
        
        # 分隔线
        pygame.draw.line(self.log_surface, self.color_dict['light-grey'], (10, 52), (self.log_panel_width - 10, 52), 1)

        # 日志内容区域
        log_start_y = 60
        # 预留给 Log 的高度 (例如总高度的 35%-40%)
        log_max_height = int(self.top_section_height * 0.48)

        # 【新增逻辑】检查是否有 final_stats
        if hasattr(self, 'final_stats') and self.final_stats is not None:
            # === 显示任务完成总结 ===
            self._draw_mission_summary(log_start_y, log_max_height)
        else:
            # === 显示实时日志 (原有逻辑) ===        
            line_height = 22
            y_offset = log_start_y

            # 获取状态数据生成日志 (保持原有逻辑)
            msgs = self._generate_log_messages() # 抽取原有逻辑到这个辅助函数，或者直接写在这里

            # 绘制文本
            for text_str, color in msgs:
                # 如果超出 Log 区域，停止绘制
                if y_offset + line_height > log_max_height:
                    break
                # 色块 + 文字
                pygame.draw.rect(self.log_surface, color, (10, y_offset + 6, 10, 10))
                text = self.medium_font.render(text_str, True, self.color_dict['black'])
                self.log_surface.blit(text, (25, y_offset))
                y_offset += line_height

        # ==========================================
        # Part B: Task Completion Board (下半部分)
        # ==========================================
        
        # 在 Log 区域下方画一条粗分割线
        board_start_y = log_max_height + 15
        pygame.draw.line(self.log_surface, self.color_dict['grey'], (5, board_start_y), (self.log_panel_width - 5, board_start_y), 2)
        
        # 调用绘制看板的辅助函数
        self._draw_task_board(start_y=board_start_y + 10)

    # 2. 新增辅助方法 _draw_mission_summary
    def _draw_mission_summary(self, start_y, max_height):
        """绘制任务完成后的统计摘要"""
        stats = self.final_stats
        
        # 1. 绘制醒目的标题
        complete_font = pygame.font.SysFont('Arial', 22, bold=True)
        title_surf = complete_font.render("MISSION COMPLETE!", True, self.color_dict['dark_green'])
        title_rect = title_surf.get_rect(center=(self.log_panel_width // 2, start_y + 20))
        self.log_surface.blit(title_surf, title_rect)
        
        # 绘制背景框
        summary_bg_rect = pygame.Rect(15, start_y + 45, self.log_panel_width - 30, 160)
        pygame.draw.rect(self.log_surface, (240, 255, 240), summary_bg_rect) # 浅绿色背景
        pygame.draw.rect(self.log_surface, self.color_dict['dark_green'], summary_bg_rect, 2)
        
        # 准备数据项
        items = [
            ("Served Count:", f"{int(stats.get('served_count', 0))}"),
            ("Total Reward:", f"{stats.get('real_reward', 0):.2f}"),
            ("Total Cost:", f"{stats.get('total_cost', 0):.2f}"),
            ("Delay Penalty:", f"{stats.get('delay_penalty', 0):.2f}"),
            ("Total Time:", f"{stats.get('episode_total_time', 0):.2f} min")
        ]
        
        # 绘制数据项
        curr_y = start_y + 60
        line_height = 28
        
        for label, value in items:
            # 标签
            label_surf = self.medium_font.render(label, True, self.color_dict['black'])
            self.log_surface.blit(label_surf, (30, curr_y))
            
            # 数值 (右对齐)
            val_surf = self.medium_font.render(value, True, self.color_dict['blue'])
            val_rect = val_surf.get_rect(topright=(self.log_panel_width - 35, curr_y))
            self.log_surface.blit(val_surf, val_rect)
            
            curr_y += line_height

    def _generate_log_messages(self):
        # 获取状态数据
        ET = self.temp_db.status_dict.get('ET')
        NT = self.temp_db.status_dict.get('NT')
        ED = self.temp_db.status_dict.get('ED')
        ND = self.temp_db.status_dict.get('ND')
        attached = self.temp_db.status_dict.get('attached_truck')
        v_coords = self.temp_db.status_dict.get('v_coord')
        
        msgs = []

        # --- 1. 生成卡车日志 ---
        for k in range(self.temp_db.num_trucks):
            status = ET[k]
            target = NT[k]
            msg = ""
            # 获取对应颜色
            color = self.truck_colors[k] if k < len(self.truck_colors) else self.color_dict['red']
            
            if status == 0: # Moving
                msg = f"Truck {k}: Moving -> Node {target}"
                if target == 0: msg += " (Depot)"
            
            elif status == 1: # Waiting
                # [核心修改] 查找该卡车在等谁
                waiting_for_drones = []
                for d_idx in range(self.temp_db.num_drones):
                    # 如果无人机归属这辆卡车，且不在车上 (ED!=3)
                    # 注意：attached[d_idx] 可能为 -1 (独立模式)，需过滤
                    if d_idx < len(attached) and attached[d_idx] == k and ED[d_idx] != 3:
                        waiting_for_drones.append(f"Drone{d_idx}")
                
                if waiting_for_drones:
                    drones_str = ", ".join(waiting_for_drones)
                    msg = f"Truck {k}: Waiting for {drones_str}..."
                else:
                    msg = f"Truck {k}: Waiting..."

            elif status == 2: # Just Served / Arrived
                msg = f"Truck {k}: Serving Node {target}"
            
            # status 3 (Idle) 也可以选择显示
            elif status == 3:
                 # 只有当不在depot或者任务未结束时才显示Idle，避免刷屏
                 if v_coords[k] != 0:
                     msg = f"Truck {k}: Idle at Node {v_coords[k]}"

            if msg:
                msgs.append((msg, color))

        # --- 2. 生成无人机日志 ---
        for k in range(self.temp_db.num_drones):
            status = ED[k]
            target = ND[k]
            att = attached[k] if k < len(attached) else -1
            color = self.drone_colors[k] if k < len(self.drone_colors) else self.color_dict['blue']
            msg = ""

            if status == 0: # Flying
                # 判断是去服务还是去汇合
                is_rendezvous = False
                target_truck = -1
                
                # 如果有归属卡车
                if att != -1 and att < self.temp_db.num_trucks:
                    # 获取卡车预期位置 (如果卡车在移动，目标是NT；如果在等待/空闲，位置是v_coord)
                    truck_target_node = NT[att] if ET[att] == 0 else v_coords[att]
                    
                    # 如果无人机飞往的目标 == 卡车的位置/目标，判定为汇合
                    if target == truck_target_node:
                        is_rendezvous = True
                        target_truck = att
                
                if is_rendezvous:
                    msg = f"Drone {k}: Returning -> Truck {target_truck}"
                else:
                    msg = f"Drone {k}: Flying -> Node {target}"
                    
            elif status == 1: # Waiting
                if att != -1:
                    # 此时无人机到了汇合点，在等卡车
                    msg = f"Drone {k}: Waiting for Truck {att}"
            
            elif status == 2: # Just Served
                msg = f"Drone {k}: Finished Service at Node {target}"

            # status 3 (On Truck) 一般不显示，或者显示 "Riding"
            
            if msg:
                msgs.append((msg, color))
            
        return msgs

    def _draw_task_board(self, start_y):
        """Draws the grid of nodes separated by Static and Dynamic requests"""
        # Get Data
        delta = self.temp_db.get_val('delta') # -1:Inactive, 0:Served, 1:Active
        road_damaged = self.temp_db.get_val('road_damaged')
        # [新增] 获取 deadline 数据
        deadlines = self.temp_db.get_val('deadline')

        if road_damaged is None:
            road_damaged = np.zeros_like(delta)
        
        # Ranges
        num_depots = self.temp_db.num_depots
        num_static = getattr(self.temp_db, 'num_static_customers', self.temp_db.num_customers)
        static_start = num_depots
        static_end = num_depots + num_static
        
        curr_y = start_y
        left_margin = 15

        # [新增] Request 部分的总标题
        main_title = self.big_font.render("Requests Overview:", True, self.color_dict['black'])
        self.log_surface.blit(main_title, (10, curr_y))
        
        # 标题下方留出空间
        curr_y += 35
        
        # Grid settings
        icon_radius = 10
        col_width = 40
        row_height = 40
        cols_per_row = (self.log_panel_width - 20) // col_width
        
        # --- 1. Static Requests Section ---
        header = self.medium_font.render("Static Requests:", True, self.color_dict['black'])
        self.log_surface.blit(header, (10, curr_y))
        curr_y += 20
        
        static_indices = list(range(static_start, static_end))
        self._draw_node_grid(static_indices, delta, road_damaged, deadlines, left_margin, curr_y, 
                           cols_per_row, col_width, row_height, icon_radius, is_dynamic=False)

        # Calculate height used by static nodes
        import math
        rows_used = math.ceil(len(static_indices) / cols_per_row) if static_indices else 0
        curr_y += max(rows_used * row_height, 20)
        
        # --- 2. Dynamic Requests Section ---
        # Separator
        pygame.draw.line(self.log_surface, self.color_dict['light-grey'], (10, curr_y), (self.log_panel_width-10, curr_y), 1)
        curr_y += 7
        
        header_dyn = self.medium_font.render("Dynamic Requests:", True, self.color_dict['black'])
        self.log_surface.blit(header_dyn, (10, curr_y))
        curr_y += 20
        
        # Filter for ACTIVE dynamic nodes (delta != -1)
        total_nodes = len(delta)
        dynamic_indices = [i for i in range(static_end, total_nodes) if delta[i] != -1]
        
        if not dynamic_indices:
            info_text = self.small_font.render("No new requests yet...", True, self.color_dict['grey'])
            self.log_surface.blit(info_text, (left_margin, curr_y))
        else:
            self._draw_node_grid(dynamic_indices, delta, road_damaged, deadlines, left_margin, curr_y, 
                               cols_per_row, col_width, row_height, icon_radius, is_dynamic=True)

    def _draw_node_grid(self, node_indices, delta, road_damaged, deadlines, start_x, start_y, 
                       cols, col_w, row_h, radius, is_dynamic):
        """Renders the grid of circle icons for nodes"""

        # 获取实际已访问完成的节点集合 (修复 Problem 2)
        visited_set = getattr(self.temp_db, 'visited_nodes', set())
        # 获取当前时间用于计算紧迫度
        current_time = getattr(self.temp_db, 'total_time', 0.0)

        for i, node_idx in enumerate(node_indices):
            # Grid math
            col = i % cols
            row = i // cols
            
            x = start_x + col * col_w
            y = start_y + row * row_h
            
            # --- 状态判断 ---
            # 【修改 3】: 只有在 visited_set 里的才算真正“完成/Served”，
            # 仅仅 delta=0 (预分配) 不算完成，不应变灰。
            is_served = (node_idx in visited_set)
            
            # 道路是否损坏
            is_damaged = (road_damaged[node_idx] == 1)

            # --- 颜色与样式逻辑 (修复 Problem 3) ---
            border_width = 1
            
            if is_served:
                # 已服务：灰色底 + 灰色字 + 后面打绿勾
                fill_color = self.color_dict['light-grey']
                border_color = self.color_dict['grey']
                text_color = self.color_dict['grey']
                
            else:
                # 未服务：保持鲜艳颜色
                text_color = self.color_dict['white']
                
                if is_damaged:
                    # 【修改 4】: 受损节点样式与地图 draw_nodes 保持一致
                    # 地图逻辑：color=red, border=red
                    fill_color = self.color_dict['red']
                    border_color = self.color_dict['red'] 
                    border_width = 2
                elif is_dynamic:
                    # 【修改 5】: 动态节点样式与地图 draw_nodes 保持一致
                    # 地图逻辑：color=active_dynamic, border=dynamic_border, width=3
                    fill_color = self.color_dict['active_dynamic']
                    border_color = self.color_dict['dynamic_border'] 
                    border_width = 3
                else:
                    # 静态节点
                    fill_color = self.color_dict['static_customer']
                    border_color = self.color_dict['black']
                    border_width = 1
            
            # --- 1. 绘制圆形底 ---
            pygame.draw.circle(self.log_surface, fill_color, (x + radius, y + radius), radius)
            
            # --- 2. 绘制边框 ---
            # 仅在非受损或受损但未服务时绘制特定边框
            pygame.draw.circle(self.log_surface, border_color, (x + radius, y + radius), radius, border_width)
            
            # --- 3. 绘制节点 ID ---
            id_text = str(node_idx)
            text_surf = self.tiny_font.render(id_text, True, text_color if not is_served else self.color_dict['black'])
            text_rect = text_surf.get_rect(center=(x + radius, y + radius))
            self.log_surface.blit(text_surf, text_rect)

            # --- [新增] 4. 在圆圈下方绘制 Deadline ---
            if not is_served:
                dl_val = deadlines[node_idx]
                time_remaining = dl_val - current_time
                # 格式化 deadline，例如 "T:120" 或直接 "120"
                dl_str = f"DDL:{dl_val:.0f}" 
                # 颜色逻辑：剩余时间少于 50 则变红，提醒注意
                if time_remaining < 30:
                    font_color = self.color_dict['red'] # 或者 'dark_red'
                else:
                    font_color = self.color_dict['black']
                dl_surf = self.tiny_font.render(dl_str, True, self.color_dict['black'])
                # 放在圆圈正下方
                dl_rect = dl_surf.get_rect(center=(x + radius, y + radius * 2 + 5))
                self.log_surface.blit(dl_surf, dl_rect)
            
            # --- 4. 覆盖层图标 (Overlay Icons) ---
            
            # Case A: Road Damaged ('X' 标记)
            # 只有未服务时，红色警告才需要醒目；或者即使服务了也保留标记但变暗
            # 这里按照地图逻辑：红叉
            # if is_damaged:
            #     cross_size = radius - 2 # 稍微小一点适应小圆圈
            #     cx, cy = x + radius, y + radius
                
            #     # 绘制 X (与地图 draw_nodes 逻辑一致，使用线条)
            #     # 为了在小图标上清晰，建议用白色或深红色，或者跟地图一样用红色(但在红底上看不清)
            #     # 地图上是因为有红底，可能看不清红线。这里为了可视性，如果未服务(红底)，我们用黑色X或白色X；
            #     # 但为了“和地图显示一样”，我们用红色线条。如果地图上也是红底红线看不清，这里也复刻该行为？
            #     # 建议：为了能看清，如果底色是红，用黑色画X。
                
            #     line_col = self.color_dict['red'] if not is_served else self.color_dict['grey']
                
            #     pygame.draw.line(self.log_surface, line_col,
            #                     (cx - cross_size, cy - cross_size),
            #                     (cx + cross_size, cy + cross_size), 2)
            #     pygame.draw.line(self.log_surface, line_col,
            #                     (cx - cross_size, cy + cross_size),
            #                     (cx + cross_size, cy - cross_size), 2)

            # Case B: Served (绿勾)
            if is_served:
                cx = x + radius
                cy = y + radius
                # 手动绘制对勾
                points = [
                    (cx - 5, cy),
                    (cx - 1, cy + 5),
                    (cx + 6, cy - 4)
                ]
                pygame.draw.lines(self.log_surface, (0, 180, 0), False, points, 3)

    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"🎮 {'Paused' if self.paused else 'Resumed'} simulation")
                elif event.key == pygame.K_RETURN and self.paused:
                    return 'step'
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse clicks for node details
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.handle_mouse_click(mouse_x, mouse_y)
        return None

    def handle_mouse_click(self, mouse_x, mouse_y):
        """处理鼠标点击，显示节点详细信息"""
        try:
            coords = self.temp_db.get_val('n_coord')
            demands = self.temp_db.get_val('n_items')
            deadlines = self.temp_db.get_val('deadline')
            delta = self.temp_db.get_val('delta')

            for i, (x, y) in enumerate(coords):
                # 修正坐标转换，与draw_nodes保持一致
                screen_x = int(self.axis_size + self.marker_size * 2 + x * self.x_mulipl)
                screen_y = int(self.axis_size + self.marker_size * 2 + (self.grid[1] - y) * self.y_mulipl)

                # Check if click is within node circle
                # 如果使用了居中布局，需要调整鼠标坐标
                if hasattr(self, 'content_start_x'):
                    adjusted_mouse_x = mouse_x - self.content_start_x
                    adjusted_mouse_y = mouse_y - self.content_start_y
                else:
                    adjusted_mouse_x = mouse_x - self.grid_padding
                    adjusted_mouse_y = mouse_y - self.grid_padding
                
                distance = np.sqrt(
                    (adjusted_mouse_x - screen_x) ** 2 + (adjusted_mouse_y - screen_y) ** 2)
                if distance <= self.marker_size + 5:
                    node_type = "Depot" if i in self.temp_db.depot_indices else "Customer"
                    status = "Visited" if delta[i] == 0 else "Unvisited"
                    print(f"\n🖱️  Clicked Node {i} ({node_type}):")
                    print(f"   Position: ({x}, {y})")
                    print(f"   Demand: {demands[i]:.2f}")
                    print(f"   Deadline: {deadlines[i]:.2f}")
                    print(f"   Status: {status}")
                    break
        except Exception as e:
            print(f"❌ Error handling mouse click: {e}")

    def visualize_step(self, episode, step, slow_down_pls=False, last_actions=None, last_rewards=None, final_stats=None):
        """可视化当前步骤 - 增强路径追踪"""
        if not self.enabled:
            return
        
        # 保存 final_stats 到实例变量
        self.final_stats = final_stats

        # 检查是否开始新的episode
        if episode != self.current_episode:
            print(f"🔄 Starting new episode {episode}, clearing previous paths")
            self.final_stats = None  # 新回合开始时清空统计数据
            
            # 清空当前episode的路径（每个episode重新开始）
            self.current_episode = episode
            for i in range(len(self.truck_paths)):
                self.truck_paths[i].clear()
                self.drone_paths[i].clear()
        
        self.current_step = step
        
        # # 记录当前位置到路径中
        # self._record_vehicle_positions()

        # 🔧 关键修复：在step=0时记录初始位置，其他时候在动作执行后记录
        if step == 0:
            # Episode开始时记录所有车辆的初始位置
            self._record_vehicle_positions()
            print(f"📍 Recorded initial positions for episode {episode}")
        else:
            # 步骤执行后记录新位置
            self._record_vehicle_positions()
        
        # 设置总训练步数用于显示
        self.total_training_steps = getattr(self, 'total_training_steps', 0) + 1

        # Create window if not exists
        if self.no_open_window:
            self.create_window()

        # Handle events
        event_result = self.handle_events()

        # Wait if paused
        while self.paused:
            event_result = self.handle_events()
            if event_result == 'step':
                break
            time.sleep(0.1)

        # Reset and draw
        self.reset_surfaces()
        self.draw_paths()  
        self.draw_nodes()
        self.draw_vehicles()
        # self.draw_legends()
        # self.draw_status_info(episode, step, last_actions, last_rewards)
        # [新增] 绘制任务日志
        self.draw_mission_log()

        # ===================== [修改] 渲染层叠逻辑 =====================
        grid_x = self.content_start_x
        grid_y = self.content_start_y
        
        # 1. 顶部：地图区域
        map_draw_x = grid_x + self.map_offset_x

        self.screen.blit(self.grid_info_surface, (map_draw_x, grid_y))
        self.screen.blit(self.grid_surface, (map_draw_x, grid_y))
        self.screen.blit(self.travel_surface, (map_draw_x, grid_y))
        
        # [新增] 绘制 Log 面板 (紧贴地图右侧)
        # grid_info_surface 的宽度包含了 axis_size，所以直接加它的宽度即可
        log_draw_x = map_draw_x + self.grid_info_surface.get_width()
        self.screen.blit(self.log_surface, (log_draw_x, grid_y))

        # 2. 底部：计算Y起始坐标
        # 注意这里使用的是 top_section_height (grid_height + axis_size)
        bottom_y = grid_y + self.top_section_height
        #底部面板的起始X (应用 bottom_offset_x 实现居中)
        bottom_draw_x = grid_x + self.bottom_offset_x
        
        # 左下：状态面板
        self.screen.blit(self.status_surface, (bottom_draw_x, bottom_y))
        
        # 右下：图例面板
        # 紧接在 Status Surface 的右边
        status_w = self.status_surface.get_width()
        self.screen.blit(self.info_surface, (bottom_draw_x + status_w, bottom_y))

        # 3. 绘制分割线
        # 线条长度等于最宽的那一部分 (self.max_content_width)
        # 这样线条会正好覆盖 较宽的那个部分 (通常是底部面板)，看起来是对齐的
        pygame.draw.line(self.screen, self.color_dict['light-grey'], 
                         (grid_x, bottom_y), 
                         (grid_x + self.max_content_width, bottom_y), 1)

                         
        # 更新屏幕显示
        pygame.display.flip()


        # # 强制等待用户输入 - 这是您要求的关键功能
        # 强制等待用户输入 - 这是您要求的关键功能
        if step == 0:
            # 第一步自动刷新一下，不停顿太久 
            time.sleep(1) # 第一步停顿1秒
        else:
            # 如果是 Mission Complete，可以稍微停顿久一点
            if final_stats is not None:
                print(f"   🏆 Episode Completed. Press ENTER to continue...")
                input()
                # time.sleep(3)
            else:
                # 正常步骤设置延迟，例如 0.5 秒（您可以根据需要调整这个数字，0.5比较适中）
                # time.sleep(1)
                print(f"   Press ENTER to continue...")
                input()

    def convert_to_img_array(self):
        """重置可视化器 - 每个episode结束时调用"""
        print(f"🔄 Episode {self.current_episode} finished, clearing paths for next episode")
        
        # 清空当前episode的所有路径轨迹
        for i in range(len(self.truck_paths)):
            self.truck_paths[i].clear()
            self.drone_paths[i].clear()

        """将当前screen转换为numpy数组，供神经网络使用"""
        if not self.no_open_window:
            self.reset_surfaces()
        print("✅ Visualizer reset completed")

        # 获取screen的像素数据
        w, h = self.screen.get_size()
        raw = pygame.surfarray.array3d(self.screen)
        # pygame使用(width, height, 3)格式，转换为标准的(height, width, 3)
        return raw.swapaxes(0, 1)

    # def reset(self):
    #     """重置可视化器"""
    #     if not self.enabled:
    #         return
    #     if not self.no_open_window:
    #         self.reset_surfaces()
    #     print("🔄 Visualizer reset")
    def reset(self):
        """Reset visualizer state for new episode."""
        print("🔄 Visualizer reset")
        self.vehicles = {}
        
        # ==========================================
        # ✅ 修复：正确清空 truck_paths 和 drone_paths
        # ==========================================
        # 这里的关键是清空实际用于绘制的列表
        if hasattr(self, 'truck_paths'):
            for path_list in self.truck_paths:
                path_list.clear()
                
        if hasattr(self, 'drone_paths'):
            for path_list in self.drone_paths:
                path_list.clear()

        # 这里的 self.paths = {} 是没用的，可以删掉或者留着防备用
        self.paths = {} 
        
        self.served_customers = set() # 重置客户颜色（变回未服务状态）
        self.final_stats = None       # 清除任务完成面板
        self.customer_service_times = {} 
        # ==========================================

        self.actions_history = []
        self.rewards_history = []
        self.current_time = 0.0
        self.episode_start_time = time.time()
        self.total_cost = 0.0
        self.current_step = 0
        
        # 强制重置当前episode计数，防止 visualize_step 中的逻辑判断失效
        # self.current_episode = -1 
        
        print("   📊 Reset episode cost tracking and CLEARED PATHS")

    def close(self):
        """关闭可视化器"""
        if not self.enabled:
            return
        if not self.no_open_window:
            pygame.quit()
            self.no_open_window = True
            print("🚪 Visualizer closed")