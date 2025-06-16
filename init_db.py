import pymysql
from dotenv import load_dotenv
import os

def init_database():
    # 加载环境变量
    load_dotenv()
    
    # 设置默认值
    host = os.getenv('MYSQL_HOST', 'localhost')
    user = os.getenv('MYSQL_USER', 'root')
    password = os.getenv('MYSQL_PASSWORD', 'root123a')  # 使用正确的密码
    database = os.getenv('MYSQL_DATABASE', 'parking_system')
    port = int(os.getenv('MYSQL_PORT', '3306'))
    
    try:
        # 连接MySQL服务器
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        
        cursor = conn.cursor()
        
        # 创建数据库
        cursor.execute(f"DROP DATABASE IF EXISTS {database}")  # 先删除已存在的数据库
        cursor.execute(f"CREATE DATABASE {database}")
        cursor.execute(f"USE {database}")
        
        print(f"数据库 '{database}' 创建成功")
        
        # 关闭连接
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"创建数据库时出错: {str(e)}")
        raise

if __name__ == "__main__":
    init_database() 