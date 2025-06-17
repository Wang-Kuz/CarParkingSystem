from datetime import datetime
from dotenv import load_dotenv
from extensions import db

load_dotenv()

class ParkingSpot(db.Model):
    __tablename__ = 'parking_spots'
    
    id = db.Column(db.Integer, primary_key=True)
    spot_number = db.Column(db.String(10), unique=True, nullable=False)  # 例如：A-01
    is_occupied = db.Column(db.Boolean, default=False)
    vehicle_plate = db.Column(db.String(20))
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime, nullable=True)
    area = db.Column(db.String(1), nullable=False)  # A, B, C, D
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ParkingRecord(db.Model):
    __tablename__ = 'parking_records'
    
    id = db.Column(db.Integer, primary_key=True)
    vehicle_plate = db.Column(db.String(20), nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime)
    spot_number = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.Float, nullable=True)  # 停车时长（小时）
    fee = db.Column(db.Float, nullable=True)  # 停车费用
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Analytics(db.Model):
    __tablename__ = 'analytics'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False)
    area = db.Column(db.String(1), nullable=False)  # A, B, C, D
    hourly_count = db.Column(db.Integer, nullable=False)  # 每小时车辆数
    created_at = db.Column(db.DateTime, default=datetime.utcnow) 