from datetime import datetime, timedelta
from models import db, ParkingRecord, ParkingSpot
from app import app
import random

def add_test_data():
    with app.app_context():
        # 清空现有记录
        ParkingRecord.query.delete()
        
        # 重置所有停车位状态
        spots = ParkingSpot.query.all()
        for spot in spots:
            spot.is_occupied = False
            spot.vehicle_plate = None
            spot.entry_time = None
        
        # 生成过去7天的测试数据
        for i in range(7):
            day = datetime.now() - timedelta(days=i)
            
            # 工作日生成更多记录
            if day.weekday() < 5:  # 周一到周五
                num_records = random.randint(30, 50)  # 工作日30-50条记录
            else:
                num_records = random.randint(15, 25)  # 周末15-25条记录
            
            # 生成当天的记录
            for j in range(num_records):
                # 根据时段分配不同的概率
                hour_weights = [1] * 24  # 基础权重
                # 早高峰 7-9点
                for h in range(7, 10):
                    hour_weights[h] = 5
                # 晚高峰 17-19点
                for h in range(17, 20):
                    hour_weights[h] = 4
                # 午饭时间 11-13点
                for h in range(11, 14):
                    hour_weights[h] = 3
                
                # 随机选择停车位（考虑区域偏好）
                area_weights = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2}
                area = random.choices(list(area_weights.keys()), 
                                   weights=list(area_weights.values()))[0]
                area_spots = [s for s in spots if s.spot_number.startswith(area)]
                spot = random.choice(area_spots)
                
                # 生成随机时间
                hour = random.choices(range(24), weights=hour_weights)[0]
                minute = random.randint(0, 59)
                entry_time = day.replace(hour=hour, minute=minute)
                
                # 生成随机停车时长（根据时段调整）
                if 7 <= hour <= 19:  # 工作时间
                    duration = random.uniform(1, 10)  # 1-10小时
                else:
                    duration = random.uniform(0.5, 3)  # 0.5-3小时
                exit_time = entry_time + timedelta(hours=duration)
                
                # 生成随机车牌号
                provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
                plate = f"{random.choice(provinces)}{random.choice('ABCDEFGHJK')}"
                for _ in range(5):
                    plate += random.choice('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
                
                # 计算费用
                if duration <= 2:
                    fee = duration * 10
                else:
                    fee = 20 + (duration - 2) * 5
                
                # 创建停车记录
                record = ParkingRecord(
                    vehicle_plate=plate,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    spot_number=spot.spot_number,
                    duration=duration,
                    fee=round(fee, 2)
                )
                db.session.add(record)
        
        # 添加一些当前正在停车的记录
        num_current = random.randint(10, 15)  # 增加当前停车数量
        current_spots = random.sample([spot.spot_number for spot in spots], num_current)
        
        for spot_number in current_spots:
            # 生成随机入场时间（1-12小时内）
            hours_ago = random.uniform(1, 12)
            entry_time = datetime.now() - timedelta(hours=hours_ago)
            
            # 生成随机车牌号
            plate = f"{random.choice(provinces)}{random.choice('ABCDEFGHJK')}"
            for _ in range(5):
                plate += random.choice('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
            
            # 更新停车位状态
            spot = ParkingSpot.query.filter_by(spot_number=spot_number).first()
            spot.is_occupied = True
            spot.vehicle_plate = plate
            spot.entry_time = entry_time
            
            # 创建停车记录
            record = ParkingRecord(
                vehicle_plate=plate,
                entry_time=entry_time,
                spot_number=spot_number
            )
            db.session.add(record)
        
        # 提交更改
        db.session.commit()
        print("测试数据添加成功！")

if __name__ == "__main__":
    add_test_data() 