# 1. ใช้ Python 3.10 เวอร์ชันกินพื้นที่น้อย (slim) เป็นฐาน
FROM python:3.10-slim

# 2. กำหนดโฟลเดอร์หลักภายใน Docker (เหมือนการสร้างโฟลเดอร์จำลอง)
WORKDIR /app

# 3. ก๊อปปี้ไฟล์ requirements.txt เข้าไปก่อน
COPY requirements.txt .

# 4. สั่งติดตั้ง Library ตามที่ระบุไว้
RUN pip install --no-cache-dir -r requirements.txt

# 5. ก๊อปปี้ไฟล์ที่เหลือทั้งหมด (app.py, ไฟล์ CSV) เข้าไปใน Docker
COPY . .

# 6. เปิดพอร์ต 8501 ให้ภายนอกสามารถเข้ามาดูเว็บได้
EXPOSE 8501

# 7. คำสั่งสุดท้ายที่จะถูกรันเมื่อเปิด Docker (เพื่อสตาร์ท Streamlit)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]