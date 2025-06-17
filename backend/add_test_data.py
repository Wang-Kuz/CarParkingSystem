from extensions import db
from models import ParkingSpot, ParkingRecord
from app import app
from datetime import datetime, timedelta
import random

def init_parking_spots():
    """初始化停车位"""
    if ParkingSpot.query.count() == 0:
        print("正在初始化停车位...")
        for area in ['A', 'B', 'C', 'D']:
            for i in range(1, 11):
                spot = ParkingSpot(spot_number=f'{area}-{i:02d}', area=area)
                db.session.add(spot)
        db.session.commit()
        print("停车位初始化完成。")
    else:
        print("停车位已存在，无需初始化。")

def add_test_data():
    """添加测试数据"""
    if ParkingRecord.query.count() > 0:
        print("测试数据已存在，跳过添加。")
        return

    print("开始添加测试数据...")

    today = datetime.now().date()
    base_time = datetime.combine(today, datetime.min.time())

    test_records = [
        {'hour': 7, 'count': 5},
        {'hour': 8, 'count': 8},
        {'hour': 9, 'count': 6},
        {'hour': 12, 'count': 4},
        {'hour': 13, 'count': 4},
        {'hour': 17, 'count': 7},
        {'hour': 18, 'count': 9},
        {'hour': 19, 'count': 5}
    ]

    # 获取全部停车位
    all_spots = ParkingSpot.query.all()

    for record in test_records:
        hour = record['hour']
        count = record['count']
        entry_time = base_time + timedelta(hours=hour)

        for i in range(count):
            spot = random.choice(all_spots)
            parking_record = ParkingRecord(
                vehicle_plate=f'测试{hour:02d}{i:02d}',
                entry_time=entry_time + timedelta(minutes=i*5),
                spot_number=spot.spot_number
            )
            db.session.add(parking_record)

            # 部分车辆已出场
            if i % 2 == 0:
                parking_record.exit_time = parking_record.entry_time + timedelta(hours=2)
                parking_record.duration = 2
                parking_record.fee = 20

    db.session.commit()
    print("测试数据添加完成。")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # 确保表存在
        init_parking_spots()
        add_test_data()