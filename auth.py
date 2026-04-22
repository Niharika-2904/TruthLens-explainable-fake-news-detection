import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_FOLDER = "database"
DB_PATH = os.path.join(DB_FOLDER, "users.db")

# DATABASE INITIALIZATION
def init_db():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    conn = sqlite3.connect(DB_PATH,timeout=10)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user'
    )
    """)

    # History table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        news TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# REGISTER USER
def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH,timeout=10)
        cursor = conn.cursor()

        hashed_password = generate_password_hash(password)

        cursor.execute("""
        INSERT INTO users (username, password, role)
        VALUES (?, ?, 'user')
        """, (username, hashed_password))

        conn.commit()
        conn.close()
        return True

    except sqlite3.IntegrityError:
        # username already exists
        return False
    
# Function to create admin panel
def create_admin():
    from werkzeug.security import generate_password_hash

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    hashed_password = generate_password_hash("admin123")

    try:
        cursor.execute("""
        INSERT INTO users (username, password, role)
        VALUES (?, ?, ?)
        """, ("admin", hashed_password, "admin"))

        conn.commit()
    except:
        pass  # Admin already exists

    conn.close()


# LOGIN USER
def login_user(username, password):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT password, role FROM users WHERE username=?
    """, (username,))

    result = cursor.fetchone()
    conn.close()

    if result:
        stored_password = result[0]
        role = result[1]

        if check_password_hash(stored_password, password):
            return role  # return role if password correct

    return None

# SAVE PREDICTION
def save_prediction(username, news, prediction, confidence):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history (username, news, prediction, confidence)
    VALUES (?, ?, ?, ?)
    """, (username, news, prediction, confidence))

    conn.commit()
    conn.close()


# GET USER HISTORY
def get_user_history(username):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT news, prediction, confidence, timestamp
    FROM history
    WHERE username = ?
    ORDER BY timestamp DESC
    """, (username,))

    rows = cursor.fetchall()
    conn.close()

    return rows