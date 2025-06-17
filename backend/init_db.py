import pymysql
from dotenv import load_dotenv
import os
from extensions import db, init_db
from models import ParkingSpot, ParkingRecord, Analytics
from flask import Flask

def init_database():
    # 加载环境变量
    load_dotenv()

    # 连接信息
    host = os.getenv('MYSQL_HOST', 'localhost')
    user = os.getenv('MYSQL_USER', 'root')
    password = os.getenv('MYSQL_PASSWORD', 'root123a')
    database = os.getenv('MYSQL_DATABASE', 'parking_system')
    port = int(os.getenv('MYSQL_PORT', '3306'))

    try:
        # 先连接 MySQL
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )

        cursor = conn.cursor()

        # 重新建库
        cursor.execute(f"DROP DATABASE IF EXISTS {database}")
        cursor.execute(f"CREATE DATABASE {database}")
        print(f"数据库 '{database}' 创建成功")

        cursor.close()
        conn.close()

        # 然后初始化表（需要在 Flask 上下文下）
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        init_db(app)

        with app.app_context():
            db.create_all()
            print("数据表创建成功")

    except Exception as e:
        print(f"初始化失败: {str(e)}")

if __name__ == "__main__":
    init_database()