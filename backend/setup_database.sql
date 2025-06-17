-- 创建数据库
CREATE DATABASE IF NOT EXISTS parking_system;

-- 使用数据库
USE parking_system;

-- 创建停车位表
CREATE TABLE IF NOT EXISTS parking_spots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    spot_number VARCHAR(10) UNIQUE,
    area CHAR(1),
    is_occupied BOOLEAN DEFAULT FALSE,
    vehicle_plate VARCHAR(20),
    entry_time DATETIME,
    exit_time DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 创建停车记录表
CREATE TABLE IF NOT EXISTS parking_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_plate VARCHAR(20),
    spot_number VARCHAR(10),
    entry_time DATETIME,
    exit_time DATETIME,
    duration FLOAT,
    fee FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 创建分析表
CREATE TABLE IF NOT EXISTS analytics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATETIME,
    area CHAR(1),
    hourly_count INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
); 