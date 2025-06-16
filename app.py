from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta, time
import os
from werkzeug.utils import secure_filename
from plate_recognition import PlateRecognizer
from parking_analytics import ParkingAnalytics
from dotenv import load_dotenv
import pymysql
from models import ParkingSpot, ParkingRecord
from sqlalchemy import func, text
from extensions import db, init_db
import time
import sqlite3
from sqlalchemy import and_

# 注册 PyMySQL 作为 MySQLdb
pymysql.install_as_MySQLdb()

# 加载环境变量
load_dotenv()

app = Flask(__name__)

# MySQL数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root123a@localhost/parking_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# 初始化数据库
init_db(app)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化车牌识别器和数据分析器
plate_recognizer = PlateRecognizer()
parking_analytics = ParkingAnalytics(db)

# 数据库连接函数
def get_db_connection():
    try:
        conn = sqlite3.connect('parking.db')
        return conn
    except Exception as e:
        print(f"数据库连接错误: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/spots', methods=['GET'])
def get_spots():
    spots = ParkingSpot.query.all()
    return jsonify([{
        'id': spot.id,
        'spot_number': spot.spot_number,
        'is_occupied': spot.is_occupied,
        'vehicle_plate': spot.vehicle_plate,
        'entry_time': spot.entry_time.isoformat() if spot.entry_time else None
    } for spot in spots])

@app.route('/api/vehicle/enter', methods=['POST'])
def vehicle_enter():
    data = request.json
    spot = ParkingSpot.query.filter_by(spot_number=data['spot_number']).first()
    
    if not spot or spot.is_occupied:
        return jsonify({'error': '停车位不可用'}), 400
    
    spot.is_occupied = True
    spot.vehicle_plate = data['vehicle_plate']
    spot.entry_time = datetime.now()
    
    record = ParkingRecord(
        vehicle_plate=data['vehicle_plate'],
        entry_time=datetime.now(),
        spot_number=data['spot_number']
    )
    
    db.session.add(record)
    db.session.commit()
    return jsonify({'message': '车辆入场成功'})

@app.route('/api/vehicle/exit', methods=['POST'])
def vehicle_exit():
    data = request.json
    spot = ParkingSpot.query.filter_by(spot_number=data['spot_number']).first()
    
    if not spot or not spot.is_occupied:
        return jsonify({'error': '停车位状态错误'}), 400
    
    record = ParkingRecord.query.filter_by(
        vehicle_plate=spot.vehicle_plate,
        exit_time=None
    ).first()
    
    if record:
        exit_time = datetime.now()
        record.exit_time = exit_time
        
        # 计算停车时长（小时）
        duration = (exit_time - record.entry_time).total_seconds() / 3600
        record.duration = duration
        
        # 计算停车费用
        # 前两小时10元/小时，之后每小时5元
        if duration <= 2:
            fee = duration * 10
        else:
            fee = 20 + (duration - 2) * 5
        record.fee = round(fee, 2)
    
    spot.is_occupied = False
    spot.vehicle_plate = None
    spot.entry_time = None
    
    db.session.commit()
    
    if record:
        return jsonify({
            'message': '车辆出场成功',
            'duration': f"{duration:.1f}小时",
            'fee': f"¥{record.fee:.2f}"
        })
    return jsonify({'message': '车辆出场成功'})

@app.route('/api/recognize-plate', methods=['POST'])
def recognize_plate():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择图片'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            plate_number, message = plate_recognizer.recognize_plate(filepath)
            os.remove(filepath)  # 识别完成后删除临时文件
            
            if plate_number:
                return jsonify({
                    'success': True,
                    'plate_number': plate_number,
                    'message': message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': message
                })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        filter_type = request.args.get('filter', 'today')
        today = datetime.now().date()
        
        if filter_type == 'today':
            start_date = today
            end_date = today + timedelta(days=1)
            chart_labels = [f"{i:02d}:00" for i in range(24)]
        elif filter_type == 'week':
            start_date = today - timedelta(days=6)
            end_date = today + timedelta(days=1)
            chart_labels = [(start_date + timedelta(days=i)).strftime('%m-%d') for i in range(7)]
        else:  # month
            start_date = today.replace(day=1)
            end_date = today + timedelta(days=1)
            chart_labels = [(start_date + timedelta(days=i)).strftime('%m-%d') 
                          for i in range((today - start_date).days + 1)]

        print(f"分析时间范围: {start_date} 到 {end_date}")

        # 获取总车位数和当前占用情况
        total_spots = ParkingSpot.query.count()
        occupied_spots = ParkingSpot.query.filter_by(is_occupied=True).count()
        current_utilization = (occupied_spots / total_spots * 100) if total_spots > 0 else 0
        print(f"总车位: {total_spots}, 当前占用: {occupied_spots}, 利用率: {current_utilization:.1f}%")

        # 统计时段内的利用率变化
        if filter_type == 'today':
            utilization_data = []
            for hour in range(24):
                hour_start = datetime.combine(today, time(hour=hour))
                hour_end = hour_start + timedelta(hours=1)
                
                # 统计该小时内的最大占用数
                occupied_count = db.session.query(ParkingSpot).filter_by(is_occupied=True).count()
                if hour >= datetime.now().hour:
                    occupied_count = 0
                
                utilization_rate = (occupied_count / total_spots * 100) if total_spots > 0 else 0
                utilization_data.append(round(utilization_rate, 1))
                print(f"小时 {hour:02d}:00 占用数: {occupied_count}, 利用率: {utilization_rate:.1f}%")
        else:
            utilization_data = []
            current_date = start_date
            while current_date <= today:
                # 获取当天的最大占用数
                day_occupied = db.session.query(ParkingSpot).filter_by(is_occupied=True).count()
                if current_date == today:
                    day_occupied = occupied_spots
                elif current_date > today:
                    day_occupied = 0
                
                utilization_rate = (day_occupied / total_spots * 100) if total_spots > 0 else 0
                utilization_data.append(round(utilization_rate, 1))
                print(f"日期 {current_date} 占用数: {day_occupied}, 利用率: {utilization_rate:.1f}%")
                current_date += timedelta(days=1)

        # 计算周转率（每个车位平均每天的使用次数）
        total_entries = ParkingRecord.query.filter(
            ParkingRecord.entry_time.between(start_date, end_date)
        ).count()
        days = max((end_date - start_date).days, 1)
        turnover_rate = (total_entries / (total_spots * days)) if total_spots > 0 else 0
        print(f"周转率: {turnover_rate:.2f} (总进场: {total_entries}, 总车位: {total_spots}, 天数: {days})")

        # 计算平均停车时长
        completed_records = ParkingRecord.query.filter(
            and_(
                ParkingRecord.entry_time.between(start_date, end_date),
                ParkingRecord.exit_time.isnot(None)
            )
        ).all()
        
        if completed_records:
            total_duration = sum(
                (record.exit_time - record.entry_time).total_seconds() / 3600 
                for record in completed_records
            )
            avg_duration = round(total_duration / len(completed_records), 1)
            print(f"平均停车时长: {avg_duration:.1f}小时 (总时长: {total_duration:.1f}, 记录数: {len(completed_records)})")
        else:
            avg_duration = 0
            print("没有完成的停车记录")

        # 计算高峰时段
        if filter_type == 'today':
            peak_query = db.session.query(
                func.hour(ParkingRecord.entry_time).label('hour'),
                func.count().label('count')
            ).filter(
                func.date(ParkingRecord.entry_time) == today
            ).group_by(
                func.hour(ParkingRecord.entry_time)
            ).order_by(
                text('count DESC')
            ).first()
            
            if peak_query and peak_query.count > 0:
                peak_hours = f"{peak_query.hour:02d}:00"
                print(f"今日高峰时段: {peak_hours} (进场数: {peak_query.count})")
            else:
                peak_hours = "--:--"
                print("今日暂无高峰时段数据")
        else:
            peak_query = db.session.query(
                func.date(ParkingRecord.entry_time).label('date'),
                func.count().label('count')
            ).filter(
                ParkingRecord.entry_time.between(start_date, end_date)
            ).group_by(
                func.date(ParkingRecord.entry_time)
            ).order_by(
                text('count DESC')
            ).first()
            
            if peak_query and peak_query.count > 0:
                peak_hours = peak_query.date.strftime('%m-%d')
                print(f"期间最繁忙日期: {peak_hours} (进场数: {peak_query.count})")
            else:
                peak_hours = "--:--"
                print("期间暂无高峰数据")

        # 获取区域统计
        area_stats = {}
        for area in ['A', 'B', 'C', 'D']:
            # 获取区域总车位数和当前占用数
            total = ParkingSpot.query.filter_by(area=area).count()
            occupied = ParkingSpot.query.filter_by(area=area, is_occupied=True).count()
            
            # 获取今日该区域的进场记录数
            traffic = ParkingRecord.query.filter(
                ParkingRecord.spot_number.like(f'{area}%'),
                func.date(ParkingRecord.entry_time) == today
            ).count()

            area_stats[area] = {
                'total': total,
                'occupied': occupied,
                'occupancy_rate': round(occupied / total * 100, 1) if total > 0 else 0,
                'today_traffic': traffic
            }
            print(f"{area}区统计: 总车位={total}, 占用={occupied}, 利用率={area_stats[area]['occupancy_rate']}%, 今日流量={traffic}")

        return jsonify({
            'success': True,
            'data': {
                'utilization_rate': round(current_utilization, 1),
                'turnover_rate': round(turnover_rate, 2),
                'avg_duration': avg_duration,
                'peak_hours': peak_hours,
                'chart_data': {
                    'labels': chart_labels,
                    'utilization': utilization_data
                },
                'area_stats': area_stats
            }
        })
    except Exception as e:
        print(f"Error in get_analytics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/income', methods=['GET'])
def get_income():
    try:
        today = datetime.now().date()
        month_start = today.replace(day=1)
        
        # 计算今日收入
        today_records = ParkingRecord.query.filter(
            func.date(ParkingRecord.exit_time) == today,
            ParkingRecord.fee.isnot(None)
        ).all()
        today_income = sum(record.fee for record in today_records)
        
        # 计算本月收入
        month_records = ParkingRecord.query.filter(
            ParkingRecord.exit_time >= month_start,
            ParkingRecord.fee.isnot(None)
        ).all()
        month_income = sum(record.fee for record in month_records)
        
        return jsonify({
            'success': True,
            'data': {
                'today_income': today_income,
                'month_income': month_income
            }
        })
    except Exception as e:
        print(f"Error in get_income: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vehicle-stats', methods=['GET'])
def get_vehicle_stats():
    try:
        today = datetime.now().date()
        past_week = [today - timedelta(days=i) for i in range(6, -1, -1)]
        
        entries_data = []
        exits_data = []
        labels = []
        
        for date in past_week:
            # 获取进场记录
            entries = ParkingRecord.query.filter(
                func.date(ParkingRecord.entry_time) == date
            ).count()
            
            # 获取出场记录
            exits = ParkingRecord.query.filter(
                func.date(ParkingRecord.exit_time) == date,
                ParkingRecord.exit_time.isnot(None)
            ).count()
            
            entries_data.append(entries)
            exits_data.append(exits)
            labels.append(date.strftime('%m-%d'))
        
        return jsonify({
            'success': True,
            'labels': labels,
            'entries_data': entries_data,
            'exits_data': exits_data
        })
    except Exception as e:
        print(f"Error in get_vehicle_stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/area-stats', methods=['GET'])
def get_area_stats():
    try:
        # 获取各区域的统计数据
        area_stats = {}
        for area in ['A', 'B', 'C', 'D']:
            # 获取区域总车位数
            total = ParkingSpot.query.filter_by(area=area).count()
            
            # 获取当前占用车位数
            occupied = ParkingSpot.query.filter_by(area=area, is_occupied=True).count()
            
            # 获取今日该区域的总车流量
            today = datetime.now().date()
            traffic = db.session.query(func.count()).filter(
                ParkingRecord.spot_number.like(f'{area}%'),
                func.date(ParkingRecord.entry_time) == today
            ).scalar()

            area_stats[area] = {
                'total': total,
                'occupied': occupied,
                'available': total - occupied,
                'occupancy_rate': round(occupied / total * 100, 2) if total > 0 else 0,
                'today_traffic': traffic or 0
            }
        
        return jsonify({
            'success': True,
            'area_stats': area_stats,
            'total_spots': sum(stats['total'] for stats in area_stats.values()),
            'total_occupied': sum(stats['occupied'] for stats in area_stats.values()),
            'total_traffic': sum(stats['today_traffic'] for stats in area_stats.values())
        })
    except Exception as e:
        print(f"Error in get_area_stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/time-distribution', methods=['GET'])
def get_time_distribution():
    try:
        print('开始获取时段分布数据')
        today = datetime.now().date()
        current_hour = datetime.now().hour
        
        print(f'当前日期: {today}, 当前小时: {current_hour}')
        
        # 获取今日每小时的进场记录
        entry_stats = db.session.query(
            func.hour(ParkingRecord.entry_time).label('hour'),
            func.count().label('count')
        ).filter(
            func.date(ParkingRecord.entry_time) == today
        ).group_by(
            func.hour(ParkingRecord.entry_time)
        ).all()
        
        print('进场统计:', entry_stats)

        # 获取今日每小时的出场记录
        exit_stats = db.session.query(
            func.hour(ParkingRecord.exit_time).label('hour'),
            func.count().label('count')
        ).filter(
            func.date(ParkingRecord.exit_time) == today,
            ParkingRecord.exit_time.isnot(None)
        ).group_by(
            func.hour(ParkingRecord.exit_time)
        ).all()
        
        print('出场统计:', exit_stats)

        # 初始化24小时的数据
        entries = [0] * 24
        exits = [0] * 24
        
        # 更新每小时的进场数据
        for hour, count in entry_stats:
            if 0 <= hour < 24:
                entries[hour] = count
        
        # 更新每小时的出场数据
        for hour, count in exit_stats:
            if 0 <= hour < 24:
                exits[hour] = count
        
        # 计算总计和当前状态
        total_entries = sum(entries)
        total_exits = sum(exits)
        current_parked = total_entries - total_exits
        
        response_data = {
            'success': True,
            'data': {
                'hourly_data': {
                    'entries': entries,
                    'exits': exits,
                    'current_hour': current_hour
                },
                'total_entries': total_entries,
                'total_exits': total_exits,
                'current_parked': current_parked
            }
        }
        
        print('返回数据:', response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_time_distribution: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ping')
def ping():
    """
    网络延迟检测API
    返回当前时间戳，前端可以计算往返时间
    """
    return jsonify({
        'timestamp': datetime.now().timestamp(),
        'status': 'ok'
    })

@app.route('/api/db-status')
def db_status():
    """
    数据库连接状态检测API
    尝试连接数据库并返回状态
    """
    try:
        conn = get_db_connection()
        if conn:
            conn.cursor().execute('SELECT 1')  # 简单的测试查询
            conn.close()
            return jsonify({
                'connected': True,
                'message': '数据库连接正常'
            })
        else:
            return jsonify({
                'connected': False,
                'message': '无法连接到数据库'
            })
    except Exception as e:
        return jsonify({
            'connected': False,
            'message': f'数据库连接错误: {str(e)}'
        })

@app.route('/api/income-details', methods=['GET'])
def get_income_details():
    try:
        # 获取查询参数
        filter_type = request.args.get('filter', 'today')
        today = datetime.now().date()
        
        if filter_type == 'today':
            # 获取今日每小时收入
            income_data = db.session.query(
                func.hour(ParkingRecord.exit_time).label('hour'),
                func.sum(ParkingRecord.fee).label('income')
            ).filter(
                func.date(ParkingRecord.exit_time) == today,
                ParkingRecord.fee.isnot(None)
            ).group_by(
                func.hour(ParkingRecord.exit_time)
            ).all()
            
            # 初始化24小时的数据
            hourly_income = [0] * 24
            for hour, income in income_data:
                if 0 <= hour < 24:
                    hourly_income[hour] = float(income or 0)
            
            return jsonify({
                'success': True,
                'data': {
                    'labels': [f"{i:02d}:00" for i in range(24)],
                    'income': hourly_income,
                    'total': sum(hourly_income)
                }
            })
            
        elif filter_type == 'month':
            # 获取本月每天收入
            month_start = today.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            income_data = db.session.query(
                func.date(ParkingRecord.exit_time).label('date'),
                func.sum(ParkingRecord.fee).label('income')
            ).filter(
                ParkingRecord.exit_time.between(month_start, month_end),
                ParkingRecord.fee.isnot(None)
            ).group_by(
                func.date(ParkingRecord.exit_time)
            ).all()
            
            # 初始化本月每天的数据
            daily_income = []
            daily_labels = []
            current_date = month_start
            while current_date <= today:
                income = 0
                for date, amount in income_data:
                    if date == current_date:
                        income = float(amount or 0)
                        break
                daily_income.append(income)
                daily_labels.append(current_date.strftime('%m-%d'))
                current_date += timedelta(days=1)
            
            return jsonify({
                'success': True,
                'data': {
                    'labels': daily_labels,
                    'income': daily_income,
                    'total': sum(daily_income)
                }
            })
            
        else:  # all
            # 获取所有收入记录
            income_data = db.session.query(
                func.date(ParkingRecord.exit_time).label('date'),
                func.sum(ParkingRecord.fee).label('income')
            ).filter(
                ParkingRecord.fee.isnot(None)
            ).group_by(
                func.date(ParkingRecord.exit_time)
            ).order_by(
                func.date(ParkingRecord.exit_time).desc()
            ).limit(30).all()  # 限制最近30天
            
            dates = []
            incomes = []
            for date, income in income_data:
                dates.append(date.strftime('%Y-%m-%d'))
                incomes.append(float(income or 0))
            
            return jsonify({
                'success': True,
                'data': {
                    'labels': dates,
                    'income': incomes,
                    'total': sum(incomes)
                }
            })
            
    except Exception as e:
        print(f"Error in get_income_details: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/daily-stats', methods=['GET'])
def get_daily_stats():
    try:
        today = datetime.now().date()
        
        # 获取今日进场和出场数量
        entries = ParkingRecord.query.filter(
            func.date(ParkingRecord.entry_time) == today
        ).count()
        
        exits = ParkingRecord.query.filter(
            func.date(ParkingRecord.exit_time) == today,
            ParkingRecord.exit_time.isnot(None)
        ).count()
        
        # 获取当前在场车辆数
        current_parked = ParkingRecord.query.filter(
            ParkingRecord.entry_time <= datetime.now(),
            ParkingRecord.exit_time.is_(None)
        ).count()
        
        # 获取总车位数和当前占用率
        total_spots = ParkingSpot.query.count()
        occupied_spots = ParkingSpot.query.filter_by(is_occupied=True).count()
        utilization_rate = (occupied_spots / total_spots * 100) if total_spots > 0 else 0
        
        # 获取今日收入
        today_income = db.session.query(func.sum(ParkingRecord.fee)).filter(
            func.date(ParkingRecord.exit_time) == today,
            ParkingRecord.fee.isnot(None)
        ).scalar() or 0
        
        return jsonify({
            'success': True,
            'data': {
                'total_entries': entries,
                'total_exits': exits,
                'current_parked': current_parked,
                'total_spots': total_spots,
                'occupied_spots': occupied_spots,
                'utilization_rate': round(utilization_rate, 1),
                'today_income': float(today_income)
            }
        })
    except Exception as e:
        print(f"Error in get_daily_stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # 初始化停车位
        if not ParkingSpot.query.first():
            print("初始化停车位...")
            # A区停车位
            for i in range(1, 11):
                spot = ParkingSpot(spot_number=f'A-{i:02d}', area='A')
                db.session.add(spot)
            
            # B区停车位
            for i in range(1, 11):
                spot = ParkingSpot(spot_number=f'B-{i:02d}', area='B')
                db.session.add(spot)
            
            # C区停车位
            for i in range(1, 11):
                spot = ParkingSpot(spot_number=f'C-{i:02d}', area='C')
                db.session.add(spot)
            
            # D区停车位
            for i in range(1, 11):
                spot = ParkingSpot(spot_number=f'D-{i:02d}', area='D')
                db.session.add(spot)
            
            db.session.commit()
            print("停车位初始化完成")

        # 添加测试数据
        if not ParkingRecord.query.first():
            print("添加测试数据...")
            # 生成今天的测试数据
            today = datetime.now().date()
            base_time = datetime.combine(today, datetime.min.time())
            
            # 模拟不同时段的进出场记录
            test_records = [
                # 早高峰
                {'hour': 7, 'count': 5},
                {'hour': 8, 'count': 8},
                {'hour': 9, 'count': 6},
                # 中午
                {'hour': 12, 'count': 4},
                {'hour': 13, 'count': 4},
                # 晚高峰
                {'hour': 17, 'count': 7},
                {'hour': 18, 'count': 9},
                {'hour': 19, 'count': 5}
            ]
            
            for record in test_records:
                hour = record['hour']
                count = record['count']
                entry_time = base_time + timedelta(hours=hour)
                
                for i in range(count):
                    # 创建进场记录
                    parking_record = ParkingRecord(
                        vehicle_plate=f'测试{hour:02d}{i:02d}',
                        entry_time=entry_time + timedelta(minutes=i*5),
                        spot_number=f'A-{(i%10 + 1):02d}'
                    )
                    db.session.add(parking_record)
                    
                    # 部分车辆已经出场
                    if i % 2 == 0:
                        parking_record.exit_time = entry_time + timedelta(hours=2, minutes=i*5)
                        parking_record.duration = 2
                        parking_record.fee = 20
            
            db.session.commit()
            print("测试数据添加完成")
    
    app.run(debug=True, port=5001) 