import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
from sqlalchemy import func
import platform

# 延迟导入 models 以避免循环导入
def get_models():
    from models import ParkingSpot, ParkingRecord
    return ParkingSpot, ParkingRecord

class ParkingAnalytics:
    def __init__(self, db):
        self.db = db
        self.ParkingSpot, self.ParkingRecord = get_models()
        # 设置中文字体
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        elif platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def calculate_occupancy_rate(self, start_date=None, end_date=None):
        """计算停车位占用率"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()

        total_spots = self.db.session.query(self.db.func.count()).select_from(self.ParkingSpot).scalar()
        records = self.ParkingRecord.query.filter(
            self.ParkingRecord.entry_time.between(start_date, end_date)
        ).all()

        # 按天统计占用率
        daily_stats = {}
        for record in records:
            entry_date = record.entry_time.date()
            exit_date = (record.exit_time or datetime.now()).date()
            
            current_date = entry_date
            while current_date <= exit_date:
                if current_date not in daily_stats:
                    daily_stats[current_date] = 0
                daily_stats[current_date] += 1
                current_date += timedelta(days=1)

        # 计算占用率
        occupancy_rates = {date: min(count / total_spots * 100, 100) 
                          for date, count in daily_stats.items()}
        
        return occupancy_rates

    def calculate_turnover_rate(self, start_date=None, end_date=None):
        """计算停车位周转率"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()

        total_spots = self.db.session.query(self.db.func.count()).select_from(self.ParkingSpot).scalar()
        total_vehicles = self.db.session.query(self.db.func.count()).select_from(self.ParkingRecord).filter(
            self.ParkingRecord.entry_time.between(start_date, end_date)
        ).scalar()

        days = (end_date - start_date).days or 1
        turnover_rate = (total_vehicles / total_spots) / days
        return turnover_rate

    def analyze_parking_duration(self):
        """分析停车时长分布"""
        records = self.ParkingRecord.query.filter(
            self.ParkingRecord.exit_time.isnot(None)
        ).all()

        durations = []
        for record in records:
            duration = (record.exit_time - record.entry_time).total_seconds() / 3600  # 转换为小时
            durations.append(duration)

        return durations

    def calculate_area_throughput(self):
        """计算各区域日吞吐量"""
        today = datetime.now().date()
        records = self.ParkingRecord.query.filter(
            func.date(self.ParkingRecord.entry_time) == today
        ).all()
        
        area_stats = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for record in records:
            area = record.spot_number[0]
            if area in area_stats:
                area_stats[area] += 1
        
        return area_stats

    def calculate_hourly_distribution(self):
        """计算时段吞吐量分布"""
        today = datetime.now().date()
        records = self.ParkingRecord.query.filter(
            func.date(self.ParkingRecord.entry_time) == today
        ).all()
        
        hourly_stats = {i: 0 for i in range(24)}
        for record in records:
            hour = record.entry_time.hour
            hourly_stats[hour] += 1
        
        return hourly_stats

    def calculate_area_status(self):
        """计算各区域车位状态"""
        spots = self.ParkingSpot.query.all()
        area_stats = {'A': {'total': 0, 'occupied': 0}, 
                     'B': {'total': 0, 'occupied': 0},
                     'C': {'total': 0, 'occupied': 0},
                     'D': {'total': 0, 'occupied': 0}}
        
        for spot in spots:
            area = spot.spot_number[0]
            if area in area_stats:
                area_stats[area]['total'] += 1
                if spot.is_occupied:
                    area_stats[area]['occupied'] += 1
        
        return area_stats

    def generate_report(self):
        """生成分析报告"""
        try:
            # 计算各项指标
            occupancy_rates = self.calculate_occupancy_rate()
            turnover_rate = self.calculate_turnover_rate()
            durations = self.analyze_parking_duration()
            area_throughput = self.calculate_area_throughput()
            hourly_distribution = self.calculate_hourly_distribution()
            area_status = self.calculate_area_status()

            # 创建图表
            plt.figure(figsize=(20, 15))

            # 1. 占用率时间序列图
            plt.subplot(331)
            dates = list(occupancy_rates.keys())
            rates = list(occupancy_rates.values())
            if dates and rates:
                plt.plot(dates, rates, marker='o')
                plt.title('停车位占用率变化')
                plt.xticks(rotation=45)
                plt.ylabel('占用率 (%)')
            else:
                plt.text(0.5, 0.5, '暂无数据', ha='center', va='center')
                plt.title('停车位占用率变化')

            # 2. 停车时长分布直方图
            plt.subplot(332)
            if durations:
                plt.hist(durations, bins=5)
                plt.title('停车时长分布')
                plt.xlabel('停车时长（小时）')
                plt.ylabel('车辆数量')
            else:
                plt.text(0.5, 0.5, '暂无数据', ha='center', va='center')
                plt.title('停车时长分布')

            # 3. 周转率
            plt.subplot(333)
            plt.bar(['周转率'], [turnover_rate])
            plt.title('日均周转率')
            plt.ylabel('周转次数/天')

            # 4. 各区域日吞吐量
            plt.subplot(334)
            areas = list(area_throughput.keys())
            throughput = list(area_throughput.values())
            plt.bar(areas, throughput)
            plt.title('各区域日吞吐量')
            plt.ylabel('车辆数量')

            # 5. 时段吞吐量分布
            plt.subplot(335)
            hours = list(hourly_distribution.keys())
            distribution = list(hourly_distribution.values())
            plt.plot(hours, distribution, marker='o')
            plt.title('时段吞吐量分布')
            plt.xlabel('小时')
            plt.ylabel('车辆数量')

            # 6. 各区域车位状态
            plt.subplot(336)
            areas = list(area_status.keys())
            occupied = [area_status[area]['occupied'] for area in areas]
            total = [area_status[area]['total'] for area in areas]
            available = [total[i] - occupied[i] for i in range(len(areas))]
            
            x = range(len(areas))
            width = 0.35
            
            plt.bar(x, occupied, width, label='已占用')
            plt.bar(x, available, width, bottom=occupied, label='空闲')
            plt.title('各区域车位状态')
            plt.xticks(x, areas)
            plt.ylabel('车位数量')
            plt.legend()

            plt.tight_layout()

            # 将图表转换为base64字符串
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            plt.close()

            return {
                'success': True,
                'occupancy_rate_avg': sum(occupancy_rates.values()) / len(occupancy_rates) if occupancy_rates else 0,
                'turnover_rate': turnover_rate,
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'area_throughput': area_throughput,
                'hourly_distribution': hourly_distribution,
                'area_status': area_status,
                'plot': img_base64
            }
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 