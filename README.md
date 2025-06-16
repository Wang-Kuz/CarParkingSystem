# 智能停车场管理系统

## 项目简介

本项目是一个基于Python Flask和现代Web技术开发的智能停车场管理系统。系统采用前后端分离架构，集成了实时监控、车辆识别、数据分析等多项功能，为停车场管理者提供全面的管理解决方案。

## 系统特点

- **实时监控**
  - 多摄像头实时监控
  - 车位状态实时显示
  - 车辆进出场实时记录
  - 系统运行状态监控

- **智能识别**
  - 车牌自动识别
  - 车辆进出场自动记录
  - 车位智能分配
  - 费用自动计算

- **数据分析**
  - 收入统计分析
  - 车位使用率分析
  - 车流量分析
  - 趋势预测

- **用户界面**
  - 响应式设计
  - 深色主题
  - 直观的操作界面
  - 流畅的交互体验

## 技术架构

### 后端技术
- Python Flask框架
- SQLite/MySQL数据库
- OpenCV图像处理
- WebSocket实时通信
- RESTful API设计

### 前端技术
- HTML5/CSS3
- 原生JavaScript
- Chart.js图表库
- WebRTC视频流
- 响应式布局

## 系统要求

### 硬件要求
- CPU：双核及以上
- 内存：4GB及以上
- 存储：50GB及以上
- 网络：100Mbps及以上

### 软件要求
- Python 3.8+
- MySQL 5.7+（可选）
- OpenCV 4.5+
- 现代浏览器（Chrome、Firefox、Safari等）

## 安装部署

1. **克隆项目**
```bash
git clone [项目地址]
cd parking-management-system
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置数据库**
```bash
# 使用默认SQLite数据库
python init_db.py

# 或配置MySQL数据库
# 修改config.py中的数据库配置
```

4. **启动服务**
```bash
python app.py
```

5. **访问系统**
```
http://localhost:5000
```

## 使用说明

### 系统初始化
1. 首次运行系统会自动创建默认停车位（A-01到A-10）
2. 管理员账号：admin
3. 默认密码：admin123

### 主要功能
1. **车辆管理**
   - 车辆入场登记
   - 车辆出场管理
   - 车位状态查看
   - 停车记录查询

2. **监控管理**
   - 实时视频监控
   - 车位状态监控
   - 异常情况报警
   - 系统状态监控

3. **数据分析**
   - 收入统计
   - 使用率分析
   - 车流量统计
   - 趋势分析

## 数据库结构

### 主要数据表
- **parking_spots**: 停车位信息表
  - spot_number: 车位编号
  - is_occupied: 占用状态
  - vehicle_plate: 车牌号
  - entry_time: 入场时间
  - exit_time: 出场时间

- **vehicle_records**: 车辆记录表
  - id: 记录ID
  - vehicle_plate: 车牌号
  - entry_time: 入场时间
  - exit_time: 出场时间
  - spot_number: 车位编号
  - fee: 停车费用

- **income_records**: 收入记录表
  - id: 记录ID
  - date: 日期
  - amount: 金额
  - record_type: 记录类型

## 安全说明

- 系统采用HTTPS加密传输
- 实现用户认证和权限管理
- 防SQL注入和XSS攻击
- 操作日志记录
- 数据定期备份

## 维护说明

### 日常维护
- 定期检查系统日志
- 监控系统运行状态
- 数据备份
- 性能优化

### 故障处理
- 检查网络连接
- 验证数据库连接
- 检查摄像头状态
- 查看错误日志

## 联系方式

- 项目负责人：[姓名]
- 邮箱：[邮箱地址]
- 电话：[联系电话]

## 版权信息

© 2024 智能停车场管理系统. All rights reserved. 